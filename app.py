
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import os
import traceback

import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "Frontend"
DB_PATH = BASE_DIR / "soundai.db"
MODEL_ROOT = BASE_DIR / "runtime_models"
MODEL_ROOT.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# Edit these only if your local dataset folder names are different.
DATASET_ALIASES = {
    ("Fan", "-6 dB"): ["-6_dB_fan", "fan_-6_dB"],
    ("Fan", "0 dB"): ["0_dB_fan", "fan_0_dB"],
    ("Fan", "+6 dB"): ["+6_dB_fan", "6_dB_fan", "fan_+6_dB"],
    ("Pump", "-6 dB"): ["-6_dB_pump", "pump_-6_dB"],
    ("Pump", "0 dB"): ["0_dB_pump", "pump_0_dB"],
    ("Pump", "+6 dB"): ["+6_dB_pump", "6_dB_pump", "pump_+6_dB"],
}

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]

FEATURE_NAMES = (
    [f"mfcc_{i}" for i in range(1, 21)]
    + [
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "zero_crossing_rate",
    ]
)

MODEL_NAMES = ["ERT", "LDA", "LGBM", "SVM", "GB"]


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            equipment TEXT NOT NULL,
            snr TEXT NOT NULL,
            file_count INTEGER NOT NULL,
            executed_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1 REAL NOT NULL,
            roc_auc REAL,
            confusion_json TEXT NOT NULL,
            labels_json TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        """
    )
    conn.commit()
    conn.close()


def dataset_path(equipment, snr):
    aliases = DATASET_ALIASES[(equipment, snr)]
    for folder in aliases:
        p = BASE_DIR / folder
        if p.exists():
            return p
    return None


def collect_wavs(root):
    if root is None:
        return []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".wav", ".wave"}
    )


def label_from_path(path):
    # Current notebook uses normal/abnormal folders.
    # Keeping this generic also supports future class folders.
    parent = path.parent.name.strip()
    return parent


def extract_features(file_path):
    y, sr = librosa.load(str(file_path), sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )
    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    )
    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)
    )
    zero_crossing_rate = np.mean(
        librosa.feature.zero_crossing_rate(y=y)
    )

    return list(mfcc_mean) + [
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate,
    ]


def build_feature_dataset(root):
    wavs = collect_wavs(root)
    records = []
    failures = []

    for i, wav in enumerate(wavs, start=1):
        try:
            features = extract_features(wav)
            row = {
                "file_name": wav.name,
                "file_path": str(wav),
                "label_name": label_from_path(wav),
            }
            row.update(dict(zip(FEATURE_NAMES, features)))
            records.append(row)
        except Exception as exc:
            failures.append({"file": str(wav), "error": str(exc)})

        if i % 250 == 0:
            print(f"Processed {i}/{len(wavs)}")

    if not records:
        raise RuntimeError("No valid WAV files could be processed.")

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df["label_name"].nunique() < 2:
        raise RuntimeError(
            "The selected dataset must contain at least two class folders."
        )

    labels = sorted(df["label_name"].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(labels)}
    df["label"] = df["label_name"].map(label_to_id)

    return df, labels, failures


def make_models(class_count):
    if LGBMClassifier is None:
        raise RuntimeError(
            "LightGBM is not installed. Run: pip install lightgbm"
        )

    common = dict(random_state=42)

    return {
        "ERT": ExtraTreesClassifier(
            n_estimators=100,
            criterion="gini",
            random_state=42,
            n_jobs=-1,
        ),
        "LDA": LinearDiscriminantAnalysis(
            solver="svd"
        ),
        "LGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            boosting_type="gbdt",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=42,
        ),
        "GB": GradientBoostingClassifier(
            criterion="friedman_mse",
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        ),
    }


def safe_auc(model, X_test, y_test, class_count):
    try:
        if not hasattr(model, "predict_proba"):
            return None

        proba = model.predict_proba(X_test)

        if class_count == 2:
            return float(
                roc_auc_score(
                    y_test,
                    proba[:, 1],
                )
            )

        return float(
            roc_auc_score(
                y_test,
                proba,
                multi_class="ovr",
                average="weighted",
            )
        )
    except Exception:
        return None


def train_and_evaluate(df, labels, dataset_key):
    feature_cols = FEATURE_NAMES
    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    # Match the existing notebook workflow:
    # SMOTE is applied only to training data.
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train, y_train
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    safe_key = (
        dataset_key
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "minus")
    )
    model_dir = MODEL_ROOT / safe_key
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.pkl")

    models = make_models(len(labels))
    output = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_scaled, y_train_smote)

        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=list(range(len(labels))),
        )

        auc = safe_auc(
            model,
            X_test_scaled,
            y_test,
            len(labels),
        )

        joblib.dump(
            model,
            model_dir / f"{model_name.lower()}_model.pkl",
        )

        output[model_name] = {
            "accuracy": round(float(accuracy) * 100, 4),
            "precision": round(float(precision) * 100, 4),
            "recall": round(float(recall) * 100, 4),
            "f1": round(float(f1) * 100, 4),
            "roc_auc": (
                round(float(auc) * 100, 4)
                if auc is not None
                else None
            ),
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        }

    return output, len(df), model_dir


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/api/register")
def register():
    data = request.get_json(force=True)

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not name or not email or not password:
        return jsonify({"error": "All fields are required."}), 400

    if len(password) < 8:
        return jsonify(
            {"error": "Password must contain at least 8 characters."}
        ), 400

    conn = db()

    try:
        conn.execute(
            """
            INSERT INTO users(name,email,password_hash,created_at)
            VALUES(?,?,?,?)
            """,
            (
                name,
                email,
                generate_password_hash(password),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify(
            {"error": "This email is already registered."}
        ), 409

    conn.close()

    return jsonify({"message": "Account created successfully."})


@app.post("/api/login")
def login():
    data = request.get_json(force=True)

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    conn = db()
    user = conn.execute(
        "SELECT id,name,email,password_hash FROM users WHERE email=?",
        (email,),
    ).fetchone()
    conn.close()

    if not user or not check_password_hash(
        user["password_hash"], password
    ):
        return jsonify(
            {"error": "Invalid email or password."}
        ), 401

    return jsonify(
        {
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
            }
        }
    )


@app.get("/api/datasets")
def datasets():
    items = []

    for (equipment, snr), aliases in DATASET_ALIASES.items():
        root = dataset_path(equipment, snr)
        count = len(collect_wavs(root)) if root else 0

        items.append(
            {
                "equipment": equipment,
                "snr": snr,
                "available": root is not None,
                "file_count": count,
            }
        )

    return jsonify(items)


@app.post("/api/run-all")
def run_all():
    data = request.get_json(force=True)

    user_id = int(data.get("user_id", 0))
    equipment = str(data.get("equipment", "")).strip()
    snr = str(data.get("snr", "")).strip()

    if (equipment, snr) not in DATASET_ALIASES:
        return jsonify({"error": "Invalid dataset selection."}), 400

    root = dataset_path(equipment, snr)

    if root is None:
        return jsonify(
            {
                "error": (
                    f"Dataset folder for {equipment} {snr} "
                    "was not found."
                )
            }
        ), 404

    dataset_key = f"{equipment} | {snr}"

    try:
        print("=" * 70)
        print(f"RUN ALL MODELS: {dataset_key}")
        print("=" * 70)

        df, labels, failures = build_feature_dataset(root)

        results, file_count, model_dir = train_and_evaluate(
            df,
            labels,
            dataset_key,
        )

        conn = db()

        cur = conn.execute(
            """
            INSERT INTO runs(
                user_id,dataset,equipment,snr,file_count,executed_at
            )
            VALUES(?,?,?,?,?,?)
            """,
            (
                user_id,
                dataset_key,
                equipment,
                snr,
                file_count,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

        run_id = cur.lastrowid

        for model_name, result in results.items():
            conn.execute(
                """
                INSERT INTO results(
                    run_id,model,accuracy,precision,recall,f1,
                    roc_auc,confusion_json,labels_json
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    model_name,
                    result["accuracy"],
                    result["precision"],
                    result["recall"],
                    result["f1"],
                    result["roc_auc"],
                    json.dumps(result["confusion_matrix"]),
                    json.dumps(result["labels"]),
                ),
            )

        conn.commit()
        conn.close()

        return jsonify(
            {
                "run_id": run_id,
                "dataset": dataset_key,
                "file_count": file_count,
                "labels": labels,
                "failed_files": len(failures),
                "results": results,
                "message": "All five models completed successfully.",
            }
        )

    except Exception as exc:
        traceback.print_exc()
        return jsonify(
            {
                "error": str(exc),
                "details": (
                    "Check the terminal running app.py for "
                    "the complete traceback."
                ),
            }
        ), 500


@app.get("/api/history")
def history():
    user_id = request.args.get("user_id", type=int)

    if not user_id:
        return jsonify({"error": "user_id is required."}), 400

    conn = db()
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.dataset,
            r.equipment,
            r.snr,
            r.file_count,
            r.executed_at,
            x.model,
            x.accuracy,
            x.precision,
            x.recall,
            x.f1,
            x.roc_auc
        FROM runs r
        JOIN results x ON x.run_id = r.id
        WHERE r.user_id=?
        ORDER BY r.id DESC, x.id ASC
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    return jsonify([dict(row) for row in rows])


@app.get("/api/run/<int:run_id>")
def run_detail(run_id):
    conn = db()
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.dataset,
            r.equipment,
            r.snr,
            r.file_count,
            r.executed_at,
            x.model,
            x.accuracy,
            x.precision,
            x.recall,
            x.f1,
            x.roc_auc,
            x.confusion_json,
            x.labels_json
        FROM runs r
        JOIN results x ON x.run_id=r.id
        WHERE r.id=?
        ORDER BY x.id ASC
        """,
        (run_id,),
    ).fetchall()
    conn.close()

    if not rows:
        return jsonify({"error": "Run not found."}), 404

    data = [dict(row) for row in rows]

    for item in data:
        item["confusion_matrix"] = json.loads(
            item.pop("confusion_json")
        )
        item["labels"] = json.loads(
            item.pop("labels_json")
        )

    return jsonify(
        {
            "run_id": run_id,
            "dataset": data[0]["dataset"],
            "file_count": data[0]["file_count"],
            "executed_at": data[0]["executed_at"],
            "results": data,
        }
    )


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    init_db()
    print("SoundAI backend running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
