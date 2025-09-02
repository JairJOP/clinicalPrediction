# ml-service/scripts/plot_shap.py
# Robust SHAP plotting that avoids beeswarm/summary_plot internals.

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
ROOT = Path(__file__).parents[1]
MODELS_DIR = ROOT / "models"

cfg_path = MODELS_DIR / "model_config.json"
with open(cfg_path, "r") as f:
    cfg = json.load(f)

MODEL_NAME = cfg.get("model_name", "logreg")
FEATURES   = cfg.get("features", [])

# Try to load saved SHAP values first
sv_bundle_path = MODELS_DIR / f"{MODEL_NAME}_shap_values_test.joblib"
X_test = None
sv = None

if sv_bundle_path.exists():
    bundle = joblib.load(sv_bundle_path)
    sv = bundle.get("shap_values", None)
    X_test = bundle.get("X_test", None)
else:
    # Fallback: try explainer + background (will be slower, but good enough to plot)
    expl_path  = MODELS_DIR / f"{MODEL_NAME}_shap_explainer.joblib"
    model_path = MODELS_DIR / f"{MODEL_NAME}_model.joblib"
    if not (expl_path.exists() and model_path.exists()):
        raise FileNotFoundError(
            "No SHAP values or explainer bundle found. Run train_models.py first."
        )
    eb = joblib.load(expl_path)
    explainer  = eb.get("explainer", None)
    background = eb.get("background", None)
    if explainer is None or background is None:
        raise RuntimeError("Explainer bundle missing 'explainer' or 'background'.")
    sv = explainer(background)
    # make a DataFrame for consistent feature names
    X_test = pd.DataFrame(background, columns=FEATURES or None)

# ---------------- Helpers ----------------
def as_array(obj):
    """Return a numpy array for SHAP values (Explanation or array-like)."""
    if hasattr(obj, "values"):  # SHAP Explanation
        arr = obj.values
    else:
        arr = obj
    arr = np.asarray(arr)
    # ensure 2D: (n_samples, n_features)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def feature_names_from(sv, X, fallback):
    # Try Explanation.feature_names
    names = getattr(sv, "feature_names", None)
    if names:
        return list(names)
    # Try X_test columns
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    # Fallback to provided list or f0..fN
    if fallback:
        return list(fallback)
    n_feat = as_array(sv).shape[1]
    return [f"f{i}" for i in range(n_feat)]

# ---------------- Prepare data ----------------
shap_vals = as_array(sv)                  # (n_samples, n_features)
feat_names = feature_names_from(sv, X_test, FEATURES)

# If X_test exists but has more columns, align to SHAP width
if isinstance(X_test, pd.DataFrame) and X_test.shape[1] != shap_vals.shape[1]:
    X_test = X_test.iloc[:, :shap_vals.shape[1]].copy()

# ---------------- Global importance (mean |SHAP|) ----------------
mean_abs = np.mean(np.abs(shap_vals), axis=0)  # (n_features,)

order = np.argsort(mean_abs)[::-1]              # descending
top_k = 15 if shap_vals.shape[1] > 15 else shap_vals.shape[1]
idx = order[:top_k]
top_feats = [feat_names[i] for i in idx]
top_vals  = mean_abs[idx]

# Save CSV for the report
csv_out = MODELS_DIR / f"{MODEL_NAME}_shap_mean_abs.csv"
pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values(
    "mean_abs_shap", ascending=False
).to_csv(csv_out, index=False)

# Bar chart (global)
plt.figure(figsize=(10, 6))
ypos = np.arange(top_k)
plt.barh(ypos, top_vals)        # no custom colors/styles per your plotting rules
plt.yticks(ypos, top_feats)
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP|")
plt.title(f"{MODEL_NAME}: Global Feature Importance")
plt.tight_layout()
out_bar = MODELS_DIR / f"{MODEL_NAME}_shap_bar.png"
plt.savefig(out_bar, dpi=200)
plt.close()
print(f"Saved: {out_bar}")
print(f"Saved: {csv_out}")

# ---------------- Instance-level explanation (first row) ----------------
# If we only have one row, this still works; if multiple, we pick the first.
inst = np.abs(shap_vals[0])               # (n_features,)
inst_order = np.argsort(inst)[::-1]
inst_idx = inst_order[:top_k]
inst_feats = [feat_names[i] for i in inst_idx]
inst_vals  = inst[inst_idx]

plt.figure(figsize=(10, 6))
ypos = np.arange(len(inst_feats))
plt.barh(ypos, inst_vals)
plt.yticks(ypos, inst_feats)
plt.gca().invert_yaxis()
plt.xlabel("|SHAP| (instance 0)")
plt.title(f"{MODEL_NAME}: Top Features for First Test Instance")
plt.tight_layout()
out_inst = MODELS_DIR / f"{MODEL_NAME}_shap_instance_bar.png"
plt.savefig(out_inst, dpi=200)
plt.close()
print(f"Saved: {out_inst}")