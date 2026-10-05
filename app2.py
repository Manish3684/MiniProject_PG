from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import numpy as np
import librosa
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)
app.secret_key = "industrial_sound_classification_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///soundai.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

PROJECT_ROOT = r"D:\PG_Project\Industrial_Sound_Classification"
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# Database
# ---------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class TestHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(40), nullable=False)
    machine = db.Column(db.String(50), nullable=False)
    snr = db.Column(db.String(20), nullable=False)
    machine_id = db.Column(db.String(50), nullable=False)
    model = db.Column(db.String(30), nullable=False)
    total_files = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1 = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()

# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------
def logged_in():
    return "user" in session

# ---------------------------------------------------------
# Dataset discovery
# Your screenshot shows:
# D:\PG_Project\Industrial_Sound_Classification\0_dB_fan
#
# This code does NOT assume Fan\0dB.
# It searches the project root for folders matching
# machine + SNR, including 0_dB_fan.
# ---------------------------------------------------------
def normalize_snr(snr):
    if snr in ("-6dB", "-6 dB", "-6"):
        return "-6"
    if snr in ("0dB", "0 dB", "0"):
        return "0"
    if snr in ("+6dB", "+6 dB", "+6", "6dB", "6 dB"):
        return "+6"
    return snr

def folder_matches(folder_name, machine, snr):
    name = folder_name.lower().replace(" ", "").replace("-", "_")
    m = machine.lower()
    s = normalize_snr(snr)

    has_machine = m in name or (m == "fan" and "fan" in name) or (m == "pump" and "pump" in name)
    if not has_machine:
        return False

    if s == "0":
        return any(x in name for x in ["0db", "0_db", "0d_b"])
    if s == "-6":
        return any(x in name for x in ["-6db", "_6db", "6db", "minus6"])
    if s == "+6":
        return any(x in name for x in ["+6db", "6db", "plus6"])

    return False

def find_dataset_path(machine, snr):
    # Exact known folder from the screenshot gets priority.
    exact_candidates = []
    s = normalize_snr(snr)

    if machine.lower() == "fan" and s == "0":
        exact_candidates.append(os.path.join(PROJECT_ROOT, "0_dB_fan"))

    # Common possible names, without assuming they exist.
    candidates = [
        f"{s}_dB_{machine.lower()}",
        f"{s}dB_{machine.lower()}",
        f"{machine.lower()}_{s}_dB",
        f"{machine}_{s}dB",
        f"{machine.lower()}_{s}db",
    ]
    for c in candidates:
        exact_candidates.append(os.path.join(PROJECT_ROOT, c))

    for p in exact_candidates:
        if os.path.isdir(p):
            return p

    # Recursive one-level search from project root.
    try:
        for item in os.listdir(PROJECT_ROOT):
            p = os.path.join(PROJECT_ROOT, item)
            if os.path.isdir(p) and folder_matches(item, machine, snr):
                return p
    except OSError:
        pass

    return None

def get_all_wav_files(folder):
    wavs = []
    if not folder or not os.path.isdir(folder):
        return wavs
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(root, f))
    return sorted(wavs)

def get_machine_id(file_path):
    # Prefer id_00/id_02/id_04/id_06 anywhere in the path.
    m = re.search(r"(id_\d+)", file_path.lower())
    return m.group(1) if m else "all"

def get_label(file_path):
    p = file_path.lower()
    # Check abnormal first in case of unusual names.
    if "abnormal" in p:
        return 1
    if "normal" in p:
        return 0
    return None

# ---------------------------------------------------------
# Feature extraction: 20 MFCC + 4 spectral = 24 features
# ---------------------------------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        if y is None or len(y) == 0:
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)

        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        x = np.concatenate([
            mfcc_mean,
            [centroid, bandwidth, rolloff, zcr]
        ]).astype(np.float64)

        return x
    except Exception as e:
        print("Feature error:", file_path, e)
        return None

# ---------------------------------------------------------
# Model discovery
# Looks through your existing output folders.
# ERT -> Fan_Module4_Output
# LDA -> Fan_Module5_Output
# Other models are discovered if their .pkl exists.
# ---------------------------------------------------------
MODEL_KEYWORDS = {
    "ERT": ["ert_model.pkl", "extra_trees_model.pkl", "ert.pkl"],
    "LDA": ["lda_model.pkl", "lda.pkl"],
    "LGBM": ["lgbm_model.pkl", "lightgbm_model.pkl", "lgbm.pkl"],
    "SVM": ["svm_model.pkl", "svm.pkl"],
    "GB": ["gb_model.pkl", "gradient_boosting_model.pkl", "gb.pkl"],
}

def find_model_file(model_name, machine, machine_id):
    # First search machine-specific output folders.
    preferred_dirs = {
        "ERT": ["Fan_Module4_Output", "Pump_Module4_Output"],
        "LDA": ["Fan_Module5_Output", "Pump_Module5_Output"],
        "LGBM": ["Fan_Module6_Output", "Pump_Module6_Output"],
        "GB": ["Fan_Module7_Output", "Pump_Module7_Output"],
        "SVM": ["Fan_Module8_Output", "Pump_Module8_Output"],
    }

    filenames = MODEL_KEYWORDS[model_name]

    for d in preferred_dirs.get(model_name, []):
        base = os.path.join(PROJECT_ROOT, d, machine_id)
        for fn in filenames:
            p = os.path.join(base, fn)
            if os.path.isfile(p):
                return p

    # Then search the whole project tree, but prefer matching machine_id.
    matches = []
    for root, _, files in os.walk(PROJECT_ROOT):
        for fn in files:
            if fn.lower() in [x.lower() for x in filenames]:
                p = os.path.join(root, fn)
                if machine_id.lower() in p.lower():
                    matches.append(p)
    if matches:
        return matches[0]

    return None

def find_scaler(machine, machine_id):
    candidates = [
        os.path.join(PROJECT_ROOT, "Fan_Module3_Output", machine_id, "scaler.pkl"),
        os.path.join(PROJECT_ROOT, "Pump_Module3_Output", machine_id, "scaler.pkl"),
        os.path.join(PROJECT_ROOT, "runtime_models", machine, machine_id, "scaler.pkl"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    for root, _, files in os.walk(PROJECT_ROOT):
        if "scaler.pkl" in [x.lower() for x in files]:
            p = os.path.join(root, "scaler.pkl")
            if machine_id.lower() in p.lower():
                return p
    return None

def load_available_models(machine, machine_id):
    loaded = {}
    paths = {}
    for model_name in ["ERT", "LDA", "LGBM", "SVM", "GB"]:
        p = find_model_file(model_name, machine, machine_id)
        if p:
            try:
                loaded[model_name] = joblib.load(p)
                paths[model_name] = p
                print(f"{model_name} loaded: {p}")
            except Exception as e:
                print(f"{model_name} load error: {e}")
        else:
            print(f"{model_name} not found for {machine}/{machine_id}")
    return loaded, paths

def load_scaler_for(machine, machine_id):
    p = find_scaler(machine, machine_id)
    if not p:
        print(f"Scaler not found for {machine}/{machine_id}")
        return None, None
    try:
        return joblib.load(p), p
    except Exception as e:
        print("Scaler load error:", e)
        return None, p

# ---------------------------------------------------------
# Evaluate ALL WAVs and ALL AVAILABLE models
# ---------------------------------------------------------
def evaluate_dataset(machine, snr):
    dataset_path = find_dataset_path(machine, snr)

    if not dataset_path:
        return {
            "error": (
                f"Dataset folder not found for {machine} / {snr}. "
                f"Project root: {PROJECT_ROOT}"
            )
        }

    wav_files = get_all_wav_files(dataset_path)
    if not wav_files:
        return {
            "error": f"No WAV files found inside: {dataset_path}"
        }

    # Group files by machine ID so each machine uses its own scaler/models.
    groups = {}
    for fp in wav_files:
        mid = get_machine_id(fp)
        groups.setdefault(mid, []).append(fp)

    final = {
        "machine": machine,
        "snr": snr,
        "dataset_path": dataset_path,
        "machine_results": {}
    }

    for machine_id, files in sorted(groups.items()):
        X, y, valid_files = [], [], []

        for fp in files:
            label = get_label(fp)
            if label is None:
                continue
            feat = extract_features(fp)
            if feat is None:
                continue
            X.append(feat)
            y.append(label)
            valid_files.append(fp)

        if not X:
            continue

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler, scaler_path = load_scaler_for(machine, machine_id)
        try:
            X_scaled = scaler.transform(X) if scaler is not None else X
        except Exception as e:
            print("Scaler transform error:", e)
            X_scaled = X

        models, model_paths = load_available_models(machine, machine_id)
        model_results = {}

        for model_name, model in models.items():
            try:
                pred = np.asarray(model.predict(X_scaled)).astype(int)

                acc = accuracy_score(y, pred)
                pre = precision_score(y, pred, zero_division=0)
                rec = recall_score(y, pred, zero_division=0)
                f1 = f1_score(y, pred, zero_division=0)
                cm = confusion_matrix(y, pred, labels=[0, 1])

                model_results[model_name] = {
                    "accuracy": round(acc * 100, 2),
                    "precision": round(pre * 100, 2),
                    "recall": round(rec * 100, 2),
                    "f1": round(f1 * 100, 2),
                    "confusion_matrix": cm.tolist(),
                    "normal": int(np.sum(pred == 0)),
                    "abnormal": int(np.sum(pred == 1)),
                    "model_path": model_paths.get(model_name, "")
                }

                db.session.add(TestHistory(
                    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    machine=machine,
                    snr=snr,
                    machine_id=machine_id,
                    model=model_name,
                    total_files=len(valid_files),
                    accuracy=acc * 100,
                    precision=pre * 100,
                    recall=rec * 100,
                    f1=f1 * 100
                ))
            except Exception as e:
                print(f"{model_name} prediction error:", e)

        final["machine_results"][machine_id] = {
            "total_files": len(valid_files),
            "normal": int(np.sum(y == 0)),
            "abnormal": int(np.sum(y == 1)),
            "scaler_path": scaler_path or "",
            "models": model_results
        }

    db.session.commit()
    return final

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("home" if logged_in() else "login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session["user"] = user.name
            session["user_id"] = user.id
            return redirect(url_for("home"))

        flash("Invalid email or password.")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        confirm = request.form["confirm_password"]

        if password != confirm:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for("register"))

        db.session.add(User(
            name=name,
            email=email,
            password=generate_password_hash(password)
        ))
        db.session.commit()

        flash("Registration successful. Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/home")
def home():
    if not logged_in():
        return redirect(url_for("login"))

    total_runs = TestHistory.query.count()
    best = db.session.query(db.func.max(TestHistory.accuracy)).scalar()

    return render_template(
        "home.html",
        user=session["user"],
        total_runs=total_runs,
        best_accuracy=round(best, 2) if best else 0
    )

@app.route("/datasets")
def datasets():
    if not logged_in():
        return redirect(url_for("login"))

    rows = []
    for machine in ["Fan", "Pump"]:
        for snr in ["-6dB", "0dB", "+6dB"]:
            path = find_dataset_path(machine, snr)
            files = get_all_wav_files(path) if path else []
            ids = sorted({get_machine_id(x) for x in files if get_machine_id(x) != "all"})
            rows.append({
                "machine": machine,
                "snr": snr,
                "path": path or "Not found",
                "files": len(files),
                "ids": ", ".join(ids) if ids else "-"
            })

    return render_template("datasets.html", datasets=rows)

@app.route("/add_dataset", methods=["GET", "POST"])
def add_dataset():
    if not logged_in():
        return redirect(url_for("login"))

    if request.method == "POST":
        machine = request.form["machine"]
        snr = request.form["snr"]
        condition = request.form["condition"]
        files = request.files.getlist("audio_files")

        # Upload to a safe app-owned folder; do not overwrite original dataset.
        folder = os.path.join(
            UPLOAD_FOLDER, machine, snr, condition
        )
        os.makedirs(folder, exist_ok=True)

        count = 0
        for f in files:
            if f and f.filename.lower().endswith(".wav"):
                safe_name = os.path.basename(f.filename)
                f.save(os.path.join(folder, safe_name))
                count += 1

        flash(f"{count} WAV file(s) uploaded.")
        return redirect(url_for("datasets"))

    return render_template("add_dataset.html")

@app.route("/testing", methods=["GET", "POST"])
def testing():
    if not logged_in():
        return redirect(url_for("login"))

    results = None
    error = None

    if request.method == "POST":
        machine = request.form["machine"]
        snr = request.form["snr"]
        results = evaluate_dataset(machine, snr)
        if "error" in results:
            error = results["error"]
            results = None

    return render_template(
        "testing.html",
        results=results,
        error=error
    )

@app.route("/history")
def history():
    if not logged_in():
        return redirect(url_for("login"))
    records = TestHistory.query.order_by(TestHistory.id.desc()).all()
    return render_template("history.html", records=records)

@app.route("/performance")
def performance():
    if not logged_in():
        return redirect(url_for("login"))
    records = TestHistory.query.order_by(TestHistory.accuracy.desc()).all()
    return render_template("performance.html", records=records)

@app.route("/comparison")
def comparison():
    if not logged_in():
        return redirect(url_for("login"))
    records = TestHistory.query.order_by(TestHistory.id.desc()).all()
    return render_template("comparison.html", records=records)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
