from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd, numpy as np, pickle, json, os, uuid, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, 'artifacts')

# ------------------- MODEL INFO (ONLY 3 MODELS) -------------------
MODEL_INFO = {
    'rf': {'name':'Random Forest','acc':97.59,'f1':0.9587,'auc':0.9991,'type':'ML'},
    'dt': {'name':'Decision Tree','acc':99.74,'f1':0.9955,'auc':0.9983,'type':'ML'},
    'gb': {'name':'Gradient Boosting','acc':94.77,'f1':0.9054,'auc':0.9885,'type':'ML'}
}

_models = {}
_scaler = None
_enc = None

# ------------------- SAFE LOADING -------------------
def load_all():
    global _models, _scaler, _enc

    for key in ['rf', 'dt', 'gb']:
        try:
            path = os.path.join(ART, f"{key}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    _models[key] = pickle.load(f)
                print(f"{key} loaded")
            else:
                print(f"{key} not found")
        except Exception as e:
            print(f"{key} load error:", e)

    try:
        with open(os.path.join(ART, "scaler.pkl"), "rb") as f:
            _scaler = pickle.load(f)
        print("scaler loaded")
    except Exception as e:
        print("scaler error:", e)

    try:
        with open(os.path.join(ART, "encoders.pkl"), "rb") as f:
            _enc = pickle.load(f)
        print("encoders loaded")
    except Exception as e:
        print("encoder error:", e)

load_all()

# ------------------- FEATURE BUILDER -------------------
def build_features(lat, lon, hour):
    return np.array([[lat, lon, hour]])

# ------------------- MODEL RUN -------------------
def run_model(model, x):
    try:
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(x)[0][1])
        else:
            return float(model.predict(x)[0])
    except:
        return 0.5

# ------------------- ROUTES -------------------

@app.route("/")
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(_models.keys()),
        "time": datetime.now().isoformat()
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}

    lat = float(data.get("lat", 42.36))
    lon = float(data.get("lon", -71.05))
    hour = int(data.get("hour", 12))
    algo = data.get("algorithm", "rf")

    if algo not in _models:
        algo = "rf"

    feat = build_features(lat, lon, hour)
    feat_s = _scaler.transform(feat) if _scaler else feat

    prob = run_model(_models[algo], feat_s)

    return jsonify({
        "algorithm": algo,
        "probability": prob,
        "risk": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    })

@app.route("/api/predict/all", methods=["POST"])
def predict_all():
    data = request.json or {}

    lat = float(data.get("lat", 42.36))
    lon = float(data.get("lon", -71.05))
    hour = int(data.get("hour", 12))

    feat = build_features(lat, lon, hour)
    feat_s = _scaler.transform(feat) if _scaler else feat

    results = {}

    for key in _models:
        prob = run_model(_models[key], feat_s)
        results[key] = {
            "probability": prob,
            "risk": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        }

    return jsonify(results)

# ------------------- RUN -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
