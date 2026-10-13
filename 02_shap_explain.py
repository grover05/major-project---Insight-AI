"""
02_shap_explain.py - Step 2: explain WHY the model flags orders as late (SHAP).

Loads the Hist Gradient Boosting model saved by 01_ml_model.py and explains its
predictions on the test set.

Reading SHAP values:
  * For each order, each feature gets a SHAP value = how much it pushed THIS order's
    late-risk up (positive) or down (negative), in log-odds units.
  * SHAP explains the MODEL's behaviour. It is not proof that a feature CAUSES
    lateness - that is what the DoWhy step (03) is for.

Run after 01_ml_model.py:   python 02_shap_explain.py
Needs: shap (pip install -U shap), plus the packages from step 1.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")                      # save plots to files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

OUT, MODELS, PROC = Path("outputs"), Path("models"), Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_SIZE = 5000
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ 1. load
bundle = joblib.load(MODELS / "hgb_late_delivery.joblib")
model = bundle["model"]
X_test = pd.read_csv(PROC / "X_test.csv")
y_test = pd.read_csv(PROC / "y_test.csv").iloc[:, 0].to_numpy()

idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
X_s, y_s = X_test.iloc[idx].reset_index(drop=True), y_test[idx]
proba = model.predict_proba(X_s)[:, 1]
print(f"Explaining {len(X_s):,} test orders | late rate in sample: {y_s.mean():.2%}")

# ------------------------------------------------------------------ 2. SHAP values
explainer = shap.TreeExplainer(model)
sv = explainer(X_s)
if sv.values.ndim == 3:                    # some models return one set per class -> keep "late"
    sv = sv[:, :, 1]


def save_fig(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ 3. global importance
shap.plots.bar(sv, max_display=15, show=False)
plt.title("Average impact on late-delivery risk (mean |SHAP|)")
save_fig("shap_bar.png")

shap.plots.beeswarm(sv, max_display=15, show=False)
save_fig("shap_beeswarm.png")             # each dot = one order; colour = feature value

importance = pd.Series(np.abs(sv.values).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
importance.to_csv(OUT / "shap_importance.csv", header=["mean_abs_shap"])

# state dummies / category dummies are split into many columns -> add them back together
groups = {}
for prefix in ["customer_state_", "seller_state_", "category_"]:
    cols = [c for c in X_s.columns if c.startswith(prefix)]
    if cols:
        total = sv.values[:, [X_s.columns.get_loc(c) for c in cols]].sum(axis=1)
        groups[prefix.rstrip("_")] = np.abs(total).mean()
grouped = pd.concat([
    importance[[c for c in X_s.columns if not c.startswith(("customer_state_", "seller_state_", "category_"))]],
    pd.Series(groups),
]).sort_values(ascending=False)
grouped.to_csv(OUT / "shap_importance_grouped.csv", header=["mean_abs_shap"])
print("\nTop factors (state/category dummies combined):")
print(grouped.head(10).round(4))

# ------------------------------------------------------------------ 4. direction of effect
print("\nDirection of effect (mean SHAP when feature is low vs high; + = raises late risk):")
rows = []
for feat in [f for f in grouped.index if f in X_s.columns][:8]:
    col = X_s[feat].to_numpy()
    s = sv.values[:, X_s.columns.get_loc(feat)]
    cut = np.nanmedian(col)
    rows.append({"feature": feat, "median": round(float(cut), 2),
                 "shap_when_low": round(float(np.nanmean(s[col <= cut])), 3),
                 "shap_when_high": round(float(np.nanmean(s[col > cut])), 3) if (col > cut).any() else np.nan})
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ 5. how the top numeric features behave
for feat in ["promised_days", "distance_km", "slack_days", "month"]:
    if feat in X_s.columns:
        shap.plots.scatter(sv[:, feat], show=False)
        plt.title(f"How {feat} changes late-delivery risk")
        save_fig(f"shap_dependence_{feat}.png")

# ------------------------------------------------------------------ 6. two individual orders
late_hit = int(np.argmax(np.where(y_s == 1, proba, -1)))        # late order the model was most sure about
ok_hit = int(np.argmin(np.where(y_s == 0, proba, 2)))           # on-time order the model was most sure about
for name, i in [("late_order", late_hit), ("ontime_order", ok_hit)]:
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    save_fig(f"shap_waterfall_{name}.png")
    print(f"\n{name}: predicted late-risk {proba[i]:.1%}, actual late = {bool(y_s[i])}")

print("\nSaved plots and tables to outputs/ (shap_*.png, shap_importance*.csv)")
