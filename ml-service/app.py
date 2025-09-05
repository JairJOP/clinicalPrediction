from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import joblib, json
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).parent
MODELS = ROOT / "models"

# Load model bundle
config_path = MODELS / "model_config.json"
with open(config_path) as f:
    cfg = json.load(f)

model_name = cfg.get("model_name", "logreg")
threshold = float(cfg.get("threshold", 0.5))
features = cfg.get("features", [])

# Model features info + age stats
model_features = features
age_mean = cfg.get("age_mean")
age_std = cfg.get("age_std")

PHQ_LONG = [
 "little_interest_or_pleasure_in_doing_things", 
 "feeling_down_depressed_or_hopeless", "trouble_falling_or_staying_asleep_or_sleeping_too_much", 
 "feeling_tired_or_having_little_energy", "poor_appetite_or_overeating", 
 "feeling_bad_about_yourself_or_that_you_are_a_failure_or_have_let_yourself_or_your_family_down", 
 "trouble_concentrating_on_things_such_as_reading_the_newspaper_or_watching_television", 
 "moving_or_speaking_so_slowly_that_other_people_could_have_noticed_or_the_opposite_being_so_fidgety_or_restless_that_you_have_been_moving_around_a_lot_more_than_usual", 
 "thoughts_that_you_would_be_better_off_dead_or_thoughts_of_hurting_yourself_in_some_way"   
]

PHQ_SHORT2LONG = {f"phq{i+1}": PHQ_LONG[i] for i in range(9)}

def normalise_payload(data: dict) -> dict:
    norm = {}

    # 1) accept both styles; convert to LONG names expected by the model
    for k, v in data.items():
        k = k.strip()
        if k in PHQ_SHORT2LONG:
            k = PHQ_SHORT2LONG[k]           # phqN -> long
        norm[k] = v

    # 2) gender -> gender_Male (0/1)
    if "gender_Male" not in norm:
        if "gender" in norm:
            g = str(norm["gender"]).strip().lower()
            norm["gender_Male"] = 1 if g in ("male", "m", "1", "true") else 0
            norm.pop("gender", None)

    # 3) handle age → age_z if the model expects it
    if "age_z" in model_features and "age" in norm:
        try:
            age_val = float(norm["age"])
            if age_std is not None and age_std > 0:
               norm["age_z"] = (age_val - age_mean) / age_std
            else:
                norm["age_z"] = 0.0
        except Exception:
            norm["age_z"] = 0.0
        norm.pop("age", None)   # remove raw age if not needed


    return norm


bundle = joblib.load(MODELS / f"{model_name}_model.joblib")
model = bundle["model"]
# feature names from bundle win if present
if "features" in bundle and bundle["features"]:
    features = bundle["features"]
    model_features = features

# Trying to load SHAP explainer
explainer = None
explainer_bundle_path = MODELS / f"{model_name}_shap_explainer.joblib"
if explainer_bundle_path.exists():
    try:
        eb = joblib.load(explainer_bundle_path)
        explainer = eb.get("explainer", None)
        background = eb.get("background", None)
    except Exception:
        explainer = None

app = Flask(__name__)
CORS(app)

def to_dataframe(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be JSON object of feature_name -> value.")
    missing = [f for f in features if f not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing[:6]}{'...' if len(missing)>6 else ''}")
    row = {f: payload[f] for f in features}
    return pd.DataFrame([row], columns=features)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": model_name, "threshold": threshold, "features": features})

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        model_name = data.get("model", "logreg")  # <-- use selected model
        model_bundle_path = MODELS / f"{model_name}_model.joblib"
        if not model_bundle_path.exists():
            raise ValueError(f"Model '{model_name}' not available")

        bundle = joblib.load(model_bundle_path)
        model = bundle["model"]
        features = bundle["features"]
        threshold = float(bundle.get("threshold", 0.5))

        data = normalise_payload(data)
        X = to_dataframe(data)

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[:, 1][0])
        else:
            proba = float(model.decision_function(X)[0])
        pred = int(proba >= threshold)
        return jsonify({
            "prediction": pred,
            "probability": proba,
            "threshold": threshold,
            "model": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/explain")
def explain():
    try:
        if explainer is None:
            return jsonify({"error": "SHAP explainer unavailable"}), 503
        data = request.get_json(force=True)
        data = normalise_payload(data)
        X = to_dataframe(data)
        sv = explainer(X)  # SHAP for this single row

        vals = np.abs(sv.values[0])
        order = np.argsort(vals)[::-1]
        top = []
        for idx in order[:10]:
            top.append({
                "feature": features[idx],
                "shap_value": float(sv.values[0][idx]),
                "abs": float(vals[idx]),
                "value": float(X.iloc[0, idx])
            })
        return jsonify({"top_features": top})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/metrics")
def metrics():
    try:
        model_name = request.args.get("model", "logreg")
        metrics_path = MODELS / f"{model_name}_metrics.json"
        if not metrics_path.exists():
            return jsonify({"error": f"Metrics for model '{model_name}' not found"}), 404
        with open(metrics_path) as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
