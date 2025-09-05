from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

import shap

# -------------------
# Paths
# -------------------
ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "phq9_train.csv"
TEST_PATH  = DATA_DIR / "phq9_test.csv"

# -------------------
# Load
# -------------------
print(f"📥 Loading:\n - {TRAIN_PATH}\n - {TEST_PATH}")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

assert "depression" in train_df.columns, "Target column 'depression' not found."

X_train = train_df.drop(columns=["depression"])
y_train = train_df["depression"].astype(int)

X_test = test_df.drop(columns=["depression"])
y_test = test_df["depression"].astype(int)

# --- safety: droping any total columns if they slipped in ---
for col in ("phq9_total", "phq9_total_norm"):
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
if col in X_test.columns:
    X_test = X_test.drop(columns=[col])    

feature_names = X_train.columns.tolist()

# -------------------
# Define models
# -------------------
models = {
    "logreg": LogisticRegression(
        solver="liblinear",  # robust for small/medium datasets
        max_iter=1000,
        class_weight="balanced"  # helps if classes slightly imbalanced
    ),
    "rf": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ),
    "xgb": XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
}

# -------------------
# Cross-validation (5-fold) on training set
# -------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = {
        "roc_auc_mean": float(np.mean(scores)),
        "roc_auc_std": float(np.std(scores)),
        "folds": [float(s) for s in scores]
    }
    print(f"CV AUC [{name}]: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# Pick the best by mean CV AUC
best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
best_model = models[best_name]
print(f"🏆 Best by CV AUC: {best_name}")

# -------------------
# Train best model on full training set
# -------------------
best_model.fit(X_train, y_train)

# -------------------
# Test-set evaluation
# -------------------
def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, zero_division=0)
    }
    return metrics, pred, proba

metrics, y_pred, y_proba = evaluate(best_model, X_test, y_test)


def threshold_sweep(y_true, y_proba, step=0.01):
    rows = []
    for t in np.arange(0.1, 0.91, step):
        yhat = (y_proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        rows.append((float(t), float(p), float(r), float(f1)))
    df = pd.DataFrame(rows, columns=["threshold","precision","recall","f1"])
    return df

ts = threshold_sweep(y_test, y_proba, step=0.01)
best_for_recall = ts.sort_values(["recall","precision","f1"], ascending=[False,False,False]).head(1)
print("\n=== Threshold sweep (top by recall) ===")
print(best_for_recall)
best_t = float(best_for_recall.iloc[0]["threshold"])

# Evaluate again using the tuned threshold
def evaluate_with_threshold(y_true, y_proba, t):
    yhat = (y_proba >= t).astype(int)
    return {
        "threshold": float(t),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, yhat).tolist(),
        "classification_report": classification_report(y_true, yhat, zero_division=0)
    }

metrics_tuned = evaluate_with_threshold(y_test, y_proba, best_t)


print("\n=== Test Metrics (best model) ===")
for k, v in metrics.items():
    if k not in ("confusion_matrix", "classification_report"):
        print(f"{k:>12}: {v:.3f}")
print("\nConfusion Matrix:\n", np.array(metrics["confusion_matrix"]))
print("\nClassification Report:\n", metrics["classification_report"])

# -------------------
# Save threshold & feature names for the API/UI
# -------------------
config = {"model_name": best_name, "threshold": best_t, "features": feature_names}
with open(OUT_DIR / "model_config.json", "w") as f:
    json.dump(config, f, indent=2)

# save the full threshold sweep table
ts.to_csv(OUT_DIR / f"{best_name}_threshold_sweep.csv", index=False)

# -------------------
# Save metrics & CV results
# -------------------
with open(OUT_DIR / f"{best_name}_metrics.json", "w") as f:
    json.dump({
        "cv": cv_results[best_name],
        "test_default": metrics,   # using 0.5 threshold
        "test_tuned": metrics_tuned  # using best_t
    }, f, indent=2)

for name, model in models.items():
    print(f"\n🚀 Training & evaluating: {name}")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump({
        "model": model,
        "features": feature_names,
        "threshold": best_t  
    }, OUT_DIR / f"{name}_model.joblib")
    print(f"✅ Saved model → {name}_model.joblib")

    # Evaluate default
    metrics, y_pred, y_proba = evaluate(model, X_test, y_test)

    # Evaluate tuned
    metrics_tuned = evaluate_with_threshold(y_test, y_proba, best_t)

    # Save metrics JSON
    with open(OUT_DIR / f"{name}_metrics.json", "w") as f:
        json.dump({
            "cv": cv_results[name],
            "test_default": metrics,
            "test_tuned": metrics_tuned
        }, f, indent=2)
    print(f"📊 Saved metrics → {name}_metrics.json")


# -------------------
# Voting ensemble
# -------------------
ensemble = VotingClassifier(
    estimators=[("lr", models["logreg"]), ("rf", models["rf"]), ("xgb", models["xgb"])],
    voting="soft", weights=[1,1,2], n_jobs=-1
)
ensemble.fit(X_train, y_train)
ens_metrics, _, _ = evaluate(ensemble, X_test, y_test)
print("\n=== Test Metrics (soft voting ensemble) ===")
for k, v in ens_metrics.items():
    if k not in ("confusion_matrix", "classification_report"):
        print(f"{k:>12}: {v:.3f}")
with open(OUT_DIR / "ensemble_metrics.json", "w") as f:
    json.dump(ens_metrics, f, indent=2)


# compute training stats for age if present
age_mean = None
age_std = None
if "age_z" in feature_names or "age" in feature_names:
    if "age_z" in train_df.columns:
        
        if "age" in train_df.columns:
            age_mean = float(train_df["age"].mean())
            age_std  = float(train_df["age"].std(ddof=0))
    elif "age" in train_df.columns:
        age_mean = float(train_df["age"].mean())
        age_std  = float(train_df["age"].std(ddof=0))

config = {
    "model_name": best_name,
    "threshold": best_t,
    "features": feature_names,
    "age_mean": age_mean,
    "age_std": age_std
}
with open(OUT_DIR / "model_config.json", "w") as f:
    json.dump(config, f, indent=2)


# -------------------
# SHAP explainer (modern API) + save values for test set
# -------------------
print("Building SHAP explainer...")
try:
    # Small background for speed if KernelExplainer is chosen 
    background = shap.sample(X_train, 200, random_state=42)
    explainer = shap.Explainer(best_model, background, feature_names=feature_names)

    sv_test = explainer(X_test)

    explainer_path = OUT_DIR / f"{best_name}_shap_explainer.joblib"
    shap_values_path = OUT_DIR / f"{best_name}_shap_values_test.joblib"
    joblib.dump({"explainer": explainer, "background": background, "features": feature_names}, explainer_path)
    joblib.dump({"shap_values": sv_test, "X_test": X_test, "y_test": y_test}, shap_values_path)

    print(f"Saved SHAP explainer → {explainer_path}")
    print(f"Saved SHAP values (test) → {shap_values_path}")
except Exception as e:
    print(f"WARN: Could not create SHAP explainer: {e}")
