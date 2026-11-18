from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime
import sqlite3, importlib.util, threading, json, os, re, traceback, shutil
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change-this-secret-key-before-deployment')
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'app_history.db'
# Save your original ML script beside app.py as industrial_sound_classification.py.
ML_SCRIPT = BASE_DIR / 'industrial_sound_classification.py'
ALLOWED_EXTENSIONS = {'.wav'}
training_state = {'running': False, 'message': 'No training run yet.', 'snr': None, 'machine': None}


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)''')
        con.execute('''CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, action TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL)''')


def log_history(action, details):
    if session.get('username'):
        with db() as con:
            con.execute('INSERT INTO history(username, action, details, created_at) VALUES(?,?,?,?)',
                        (session['username'], action, str(details), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def load_ml_module():
    if not ML_SCRIPT.is_file():
        raise FileNotFoundError(f'Original ML script not found: {ML_SCRIPT}. Save your existing code with this filename beside app.py.')
    spec = importlib.util.spec_from_file_location('industrial_sound_original', ML_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_snr(path):
    text = str(path).lower().replace('+', 'plus').replace('-', 'minus')
    if re.search(r'(?<!\d)minus\s*6\s*_?db|(?<!\d)minus_?6db|(?<!\d)-6\s*_?db', text) or '-6db' in str(path).lower() or '-6_db' in str(path).lower():
        return '-6 dB'
    if re.search(r'(?<!\d)(?:plus\s*6|\+\s*6|6)\s*_?db', text) or '+6db' in str(path).lower() or '+6_db' in str(path).lower():
        return '+6 dB'
    if re.search(r'(?<!\d)0\s*_?db', text) or '0db' in str(path).lower():
        return '0 dB'
    return 'Unknown'


def get_metrics():
    metrics_file = None
    try:
        m = load_ml_module()
        metrics_file = m.OUTPUT_DIR / 'frontend_metrics.json'
        if metrics_file.exists():
            return json.loads(metrics_file.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {'best_model': 'Not trained yet', 'best_accuracy': None, 'models': [], 'snr': {}, 'dataset_counts': {}, 'last_run': None}


def train_worker(selected_snr):
    training_state['running'] = True
    training_state['message'] = f'Training five models for Fan and Pump at {selected_snr}...'
    try:
        import joblib
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
        m = load_ml_module()
        dataframe = m.load_dataset()
        snr_values = dataframe['audio_path'].map(detect_snr)
        subset = dataframe.loc[snr_values == selected_snr].copy()
        expected = ['Fan_Normal', 'Fan_Abnormal', 'Pump_Normal', 'Pump_Abnormal']
        if subset.empty:
            raise ValueError(f'No labelled WAV files found for {selected_snr}. Check the dB folder names.')
        missing = set(expected) - set(subset['label'].unique())
        if missing:
            raise ValueError(f'{selected_snr} needs all four classes: {", ".join(sorted(missing))}. Add missing Fan/Pump Normal/Abnormal recordings.')
        counts = subset['label'].value_counts()
        n_splits = min(5, int(counts.min()))
        if n_splits < 2:
            raise ValueError('At least two labelled WAV files for each condition are required to evaluate accuracy.')
        X = subset[m.FEATURE_NAMES].to_numpy(dtype=float)
        y_text = subset['label'].to_numpy()
        encoder = m.LabelEncoder()
        y = encoder.fit_transform(y_text)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=m.RANDOM_STATE)
        models = m.get_models()
        # Ensure five algorithms are available: install LightGBM if it was skipped.
        if 'LightGBM' not in models:
            try:
                from lightgbm import LGBMClassifier
                models['LightGBM'] = LGBMClassifier(n_estimators=150, learning_rate=0.05,
                    class_weight='balanced', random_state=m.RANDOM_STATE, verbosity=-1, n_jobs=-1)
            except ImportError:
                raise RuntimeError('The 5th model (LightGBM) is unavailable. Run: python -m pip install lightgbm')
        results = []
        artifacts = {}
        for model_name, base_model in models.items():
            oof = np.full(len(y), -1, dtype=int)
            for train_idx, valid_idx in cv.split(X, y):
                X_train_ready, y_train_ready, imputer, scaler = m.prepare_training_data(
                    X[train_idx], y[train_idx], apply_smote=False)
                model = clone(base_model)
                model.fit(X_train_ready, y_train_ready)
                X_valid_ready = scaler.transform(imputer.transform(X[valid_idx]))
                oof[valid_idx] = model.predict(X_valid_ready).astype(int)
            acc = float(accuracy_score(y, oof))
            cm = confusion_matrix(y, oof, labels=np.arange(len(encoder.classes_)))
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            cm_dir = BASE_DIR / 'static' / 'confusion_matrices'
            cm_dir.mkdir(parents=True, exist_ok=True)
            safe_snr_name = {'-6 dB': 'minus6db', '0 dB': '0db', '+6 dB': 'plus6db'}[selected_snr]
            safe_model_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', model_name).lower()
            cm_filename = f'cm_{safe_snr_name}_{safe_model_name}.png'
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_).plot(
                ax=ax, cmap='Blues', values_format='d', colorbar=False, xticks_rotation=30)
            ax.set_title(f'{model_name} — {selected_snr}')
            fig.tight_layout()
            fig.savefig(cm_dir / cm_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            results.append({
                'Model': model_name, 'Accuracy': acc,
                'Macro_Precision': float(precision_score(y, oof, average='macro', zero_division=0)),
                'Macro_Recall': float(recall_score(y, oof, average='macro', zero_division=0)),
                'Macro_F1': float(f1_score(y, oof, average='macro', zero_division=0)),
                'confusion_matrix': cm.tolist(),
                'class_names': encoder.classes_.tolist(),
                'confusion_image': f'confusion_matrices/{cm_filename}'
            })
            # Fit a final deployment model using all samples for this selected dB/machine.
            X_all_ready, y_all_ready, final_imputer, final_scaler = m.prepare_training_data(
                X, y, apply_smote=False)
            final_model = clone(base_model)
            final_model.fit(X_all_ready, y_all_ready)
            artifacts[model_name] = {'model': final_model, 'imputer': final_imputer,
                'scaler': final_scaler, 'label_encoder': encoder,
                'feature_names': m.FEATURE_NAMES, 'sample_rate': m.SAMPLE_RATE, 'n_mfcc': m.N_MFCC}
        safe_snr = {'-6 dB': 'minus6db', '0 dB': '0db', '+6 dB': 'plus6db'}[selected_snr]
        key = f'{safe_snr}_both'
        m.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_file = m.MODEL_DIR / f'five_models_{key}.joblib'
        joblib.dump(artifacts, model_file)
        results.sort(key=lambda x: (x['Accuracy'], x['Macro_F1']), reverse=True)
        metrics_path = m.OUTPUT_DIR / 'frontend_metrics.json'
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
        except Exception:
            metrics = {}
        group_key = f'{selected_snr}|Both'
        combos = metrics.get('combinations', {})
        combos[group_key] = {
            'snr': selected_snr, 'machine': 'Both Fan and Pump',
            'count': int(len(subset)), 'class_counts': subset['label'].value_counts().to_dict(),
            'models': results, 'best_model': results[0]['Model'],
            'best_accuracy': results[0]['Accuracy'], 'model_file': str(model_file),
            'class_names': encoder.classes_.tolist(),
            'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation': f'{n_splits}-fold stratified cross-validation'
        }
        metrics['combinations'] = combos
        metrics['active_snr'] = selected_snr
        metrics['active_machine'] = 'Both Fan and Pump'
        metrics['best_model'] = results[0]['Model']
        metrics['best_accuracy'] = results[0]['Accuracy']
        metrics['models'] = results
        metrics['total_files'] = int(len(subset))
        metrics['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['note'] = f'Five models evaluated for Fan Normal, Fan Abnormal, Pump Normal and Pump Abnormal at {selected_snr}.'
        m.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        training_state['message'] = f'{selected_snr} / Fan and Pump: all five models completed. Best: {results[0]["Model"]} ({results[0]["Accuracy"]:.2%}).'
    except Exception as exc:
        training_state['message'] = f'Training failed: {exc}'
        traceback.print_exc()
    finally:
        training_state['running'] = False


@app.route('/')
def index():
    return redirect(url_for('home') if session.get('username') else url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if len(username) < 3:
            flash('Username must contain at least 3 characters.', 'error')
        elif len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'[a-z]', password) or not re.search(r'\d', password) or not re.search(r'[^A-Za-z0-9]', password):
            flash('Password needs 8+ characters, uppercase, lowercase, number, and special character.', 'error')
        else:
            try:
                with db() as con:
                    con.execute('INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)',
                                (username, generate_password_hash(password), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                flash('Registration successful. Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('That username already exists.', 'error')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        with db() as con:
            user = con.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session.clear(); session['username'] = username
            log_history('Login', 'Signed in successfully')
            return redirect(url_for('home'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    if session.get('username'): log_history('Logout', 'Signed out')
    session.clear()
    return redirect(url_for('login'))


def login_required():
    if not session.get('username'):
        return redirect(url_for('login'))
    return None


@app.route('/home')
def home():
    check = login_required()
    if check: return check
    metrics = get_metrics()
    return render_template('home.html', metrics=metrics, state=training_state)


@app.route('/dataset')
def dataset():
    check = login_required()
    if check: return check
    try:
        m = load_ml_module(); root = m.DATASET_ROOT
        counts = {k: 0 for k in ['-6 dB', '0 dB', '+6 dB', 'Unknown']}
        class_counts = {k: 0 for k in m.CLASS_NAMES}
        snr_class_counts = {snr: {label: 0 for label in m.CLASS_NAMES} for snr in ['-6 dB', '0 dB', '+6 dB']}
        files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.wav'] if root.exists() else []
        for p in files:
            snr_value = detect_snr(p)
            counts[snr_value] += 1
            label = m.identify_class(p)
            if label in class_counts:
                class_counts[label] += 1
                if snr_value in snr_class_counts:
                    snr_class_counts[snr_value][label] += 1
        root_path = str(root)
    except Exception as exc:
        counts = {'-6 dB': 0, '0 dB': 0, '+6 dB': 0, 'Unknown': 0}; class_counts = {}; snr_class_counts = {}; root_path = str(exc)
    return render_template('dataset.html', counts=counts, class_counts=class_counts, snr_class_counts=snr_class_counts, root_path=root_path)


@app.route('/add-dataset', methods=['GET', 'POST'])
def add_dataset():
    check = login_required()
    if check: return check
    if request.method == 'GET':
        return render_template('add_dataset.html')
    try:
        m = load_ml_module()
        snr = request.form.get('snr', '')
        machine = request.form.get('machine', '')
        status = request.form.get('status', '')
        uploaded = request.files.getlist('audio_files')
        if snr not in ['-6 dB', '0 dB', '+6 dB'] or machine not in ['Fan', 'Pump'] or status not in ['Normal', 'Abnormal']:
            flash('Choose a valid dB category, machine, and condition.', 'error'); return redirect(url_for('dataset'))
        uploaded = [f for f in uploaded if f and f.filename]
        if not uploaded:
            flash('Select at least one WAV file.', 'error'); return redirect(url_for('dataset'))
        snr_folder = {'-6 dB': '-6_dB', '0 dB': '0_dB', '+6 dB': '+6_dB'}[snr]
        target = m.DATASET_ROOT / snr_folder / f'{machine}_{status}'
        target.mkdir(parents=True, exist_ok=True)
        added = 0
        for f in uploaded:
            name = secure_filename(f.filename)
            if Path(name).suffix.lower() != '.wav': continue
            dest = target / name
            if dest.exists(): dest = target / f'{Path(name).stem}_{datetime.now().strftime("%H%M%S%f")}.wav'
            f.save(dest); added += 1
        if added:
            log_history('Dataset added', f'{added} WAV file(s), {snr}, {machine}_{status}, {target}')
            flash(f'Added {added} WAV file(s) to {target}. Run model training to refresh metrics.', 'success')
        else: flash('No WAV files were uploaded. Only .wav files are accepted.', 'error')
    except Exception as exc: flash(str(exc), 'error')
    return redirect(url_for('dataset'))


@app.route('/test', methods=['GET', 'POST'])
def test_model():
    check = login_required()
    if check: return check
    if request.method == 'POST':
        mode = request.form.get('mode', 'evaluate')
        snr = request.form.get('snr', '-6 dB')
        if snr not in ['-6 dB', '0 dB', '+6 dB']:
            flash('Choose a valid dB category.', 'error')
            return redirect(url_for('test_model'))
        if mode == 'predict':
            try:
                uploaded = request.files.get('audio_file')
                if not uploaded or not uploaded.filename:
                    raise ValueError('Please click Choose File and upload one WAV audio file.')
                safe_name = secure_filename(uploaded.filename)
                if Path(safe_name).suffix.lower() != '.wav':
                    raise ValueError('Only .wav audio files are accepted.')
                upload_dir = BASE_DIR / 'uploaded_test_audio'
                upload_dir.mkdir(parents=True, exist_ok=True)
                saved = upload_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_{safe_name}'
                uploaded.save(saved)
                audio_path = str(saved)
                import joblib
                m = load_ml_module()
                safe_snr = {'-6 dB': 'minus6db', '0 dB': '0db', '+6 dB': 'plus6db'}[snr]
                model_file = m.MODEL_DIR / f'five_models_{safe_snr}_both.joblib'
                if not model_file.is_file():
                    raise FileNotFoundError(
                        f'No trained model bundle exists for {snr}. First train Fan + Pump for this dB category.'
                    )
                artifacts = joblib.load(model_file)
                features = m.extract_features(audio_path)
                if features is None:
                    raise ValueError('Could not read this WAV file or extract its audio features.')
                Xnew = np.asarray(features, dtype=float).reshape(1, -1)
                predictions = []
                for model_name, artifact in artifacts.items():
                    transformed = artifact['imputer'].transform(Xnew)
                    transformed = artifact['scaler'].transform(transformed)
                    pred_id = artifact['model'].predict(transformed)
                    label = artifact['label_encoder'].inverse_transform(pred_id)[0]
                    predictions.append({'model': model_name, 'prediction': label})
                result = {'predictions': predictions, 'audio_path': safe_name, 'snr': snr, 'mode': 'predict'}
                log_history('New audio prediction', f'{safe_name} | {snr} | ' + '; '.join(f"{p['model']}={p['prediction']}" for p in predictions))
            except Exception as exc:
                result = {'error': str(exc), 'snr': snr, 'mode': 'predict'}
            return render_template('test.html', result=result, state=training_state, selected_snr=snr)

        if training_state['running']:
            flash('A training job is already running. Please wait.', 'error')
            return redirect(url_for('test_model', snr=snr))
        training_state['snr'] = snr
        training_state['machine'] = 'Both'
        training_state['running'] = True
        training_state['message'] = f'Starting five-model evaluation for Fan and Pump at {snr}...'
        threading.Thread(target=train_worker, args=(snr,), daemon=True).start()
        log_history('Model training started', f'Automatic five-model evaluation for Fan and Pump at {snr}')
        return redirect(url_for('test_model', snr=snr))

    snr = request.args.get('snr', training_state.get('snr') or '-6 dB')
    result = None
    if training_state.get('running') and training_state.get('snr') == snr:
        result = {'running': True, 'message': training_state.get('message', 'Training models...'), 'snr': snr, 'mode': 'evaluate'}
    elif training_state.get('snr') == snr and training_state.get('message', '').startswith('Training failed:'):
        result = {'error': training_state['message'], 'snr': snr, 'mode': 'evaluate'}
    else:
        metrics = get_metrics()
        combo = metrics.get('combinations', {}).get(f'{snr}|Both')
        if combo:
            result = {'snr': snr, 'combo': combo, 'message': training_state.get('message', ''), 'mode': 'evaluate'}
    return render_template('test.html', result=result, state=training_state, selected_snr=snr)


@app.route('/performance')
def performance():
    check = login_required()
    if check: return check
    metrics = get_metrics()
    selected = request.args.get('snr', 'All')
    return render_template('performance.html', metrics=metrics, selected=selected)


@app.route('/history')
def history():
    check = login_required()
    if check: return check
    with db() as con:
        rows = con.execute('SELECT action, details, created_at FROM history WHERE username=? ORDER BY id DESC LIMIT 500', (session['username'],)).fetchall()
    return render_template('history.html', rows=rows)


@app.route('/train', methods=['POST'])
def train():
    check = login_required()
    if check: return check
    selected_snr = request.form.get('snr', '-6 dB')
    if selected_snr not in ['-6 dB', '0 dB', '+6 dB']:
        flash('Choose a valid dB category.', 'error')
    elif training_state['running']:
        flash('Training is already running.', 'error')
    else:
        training_state['snr'] = selected_snr
        training_state['machine'] = 'Both'
        training_state['running'] = True
        threading.Thread(target=train_worker, args=(selected_snr,), daemon=True).start()
        log_history('Model training started', f'Training five models for Fan and Pump at {selected_snr}')
        flash(f'Training started for Fan and Pump at {selected_snr}.', 'success')
    return redirect(url_for('home'))

init_db()
if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
