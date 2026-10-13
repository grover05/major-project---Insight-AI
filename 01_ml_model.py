
"""
01_ml_model.py - Step 1: predict late deliveries and report train/test scores.

Business question: can we flag orders likely to be delivered late, using only
information available at purchase time?

Run from the project folder:   python 01_ml_model.py
Needs: pandas numpy scikit-learn matplotlib joblib   (scikit-learn >= 1.2)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from data_prep import build_model_table, get_X_y

OUT, MODELS, PROC = Path("outputs"), Path("models"), Path("data/processed")
for folder in (OUT, MODELS, PROC):
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ 1. data
df = build_model_table()
df.to_csv(PROC / "model_table.csv", index=False)      # reused by the DoWhy step
X, y = get_X_y(df)
print(f"Orders: {len(df):,} | features: {X.shape[1]} | late rate: {y.mean():.2%}")


# ------------------------------------------------------------------ 2. splits
def random_split(X, y):
    """80/20, stratified so both sets keep the same late rate."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def time_split(X, y, df):
    """Train on the earliest 80% of orders, test on the latest 20% (more realistic)."""
    order = df["order_purchase_timestamp"].sort_values().index
    cut = int(len(order) * 0.8)
    tr, te = order[:cut], order[cut:]
    return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]


splits = {
    "random split": random_split(X, y),
    "time-based split": time_split(X, y, df),
}


# ------------------------------------------------------------------ 3. models
def make_models():
    return {
        "Baseline (always predicts prior)": DummyClassifier(strategy="prior"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300,
            class_weight="balanced", random_state=42),
    }


def score(model, X_, y_):
    proba = model.predict_proba(X_)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_, pred),
        "precision_late": precision_score(y_, pred, zero_division=0),
        "recall_late": recall_score(y_, pred, zero_division=0),
        "f1_late": f1_score(y_, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_, proba),
    }


# ------------------------------------------------------------------ 4. train + evaluate
rows, saved = [], {}
for split_name, (X_tr, X_te, y_tr, y_te) in splits.items():
    medians = X_tr.median()                 # fill missing values using TRAIN data only
    X_tr, X_te = X_tr.fillna(medians), X_te.fillna(medians)

    for model_name, model in make_models().items():
        model.fit(X_tr, y_tr)
        for set_name, (Xs, ys) in {"train": (X_tr, y_tr), "test": (X_te, y_te)}.items():
            rows.append({"split": split_name, "model": model_name, "set": set_name,
                         **score(model, Xs, ys)})
        if split_name == "random split":
            saved[model_name] = (model, X_tr, X_te, y_tr, y_te, medians)

results = pd.DataFrame(rows)
results.to_csv(OUT / "ml_metrics.csv", index=False)

pd.set_option("display.width", 200)
for split_name in splits:
    print(f"\n===== {split_name} =====")
    print(results[results["split"] == split_name]
          .drop(columns="split").round(3).to_string(index=False))

# ------------------------------------------------------------------ 5. feature importance (RF)
rf, X_tr, X_te, y_tr, y_te, medians = saved["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
importance.to_csv(OUT / "rf_feature_importance.csv", header=["importance"])
print("\nTop 10 features (Random Forest):")
print(importance.head(10).round(4))

ax = importance.head(15).iloc[::-1].plot.barh(figsize=(7, 5))
ax.set_title("Random Forest - top 15 features for predicting late delivery")
plt.tight_layout()
plt.savefig(OUT / "rf_feature_importance.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 6. save for the SHAP step
hgb = saved["Hist Gradient Boosting"][0]    # stronger model -> this is the one SHAP will explain
joblib.dump({"model": hgb, "columns": list(X_tr.columns), "medians": medians},
            MODELS / "hgb_late_delivery.joblib")
X_te.to_csv(PROC / "X_test.csv", index=False)
y_te.to_csv(PROC / "y_test.csv", index=False)
print("\nSaved: models/rf_late_delivery.joblib, data/processed/X_test.csv, y_test.csv, outputs/*")
