"""
03_dowhy_causal.py - Step 3: does a late delivery CAUSE worse customer reviews?

SHAP (step 2) told us what the late-delivery MODEL relies on. This step asks a
different question: if an order arrives late, how much does the review score drop
because of that - after accounting for other things that differ between late and
on-time orders (distance, product type, state, season, ...)?

DoWhy workflow:  1) model the assumptions  2) identify the effect
                 3) estimate it            4) try to break it (refutation tests)

  Treatment : is_late          (1 = delivered after the promised date)
  Outcome   : review_score     (1-5)   and   low_review (1-2 stars)
  Confounders (affect BOTH lateness and reviews): distance, freight, price, weight,
      volume, items, promised days, shipping slack, season, state, product category.
  NOT included on purpose: delay_days and delivery dates - those are consequences of
  lateness, and adjusting for them would hide part of the effect we want to measure.

Run after 01_ml_model.py (it saves data/processed/model_table.csv):
    python 03_dowhy_causal.py
Needs: dowhy (python -m pip install dowhy), plus packages from the earlier steps.
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dowhy import CausalModel

from data_prep import CAT_FEATURES, NUM_FEATURES

warnings.filterwarnings("ignore")
OUT, PROC = Path("outputs"), Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
N_SIM = 20                                    # refutation simulations (raise for a final run)

# ------------------------------------------------------------------ 1. data
df = pd.read_csv(PROC / "model_table.csv")
df = df.dropna(subset=["review_score"]).reset_index(drop=True)      # ~0.7% have no review
df["low_review"] = (df["review_score"] <= 2).astype(int)

X = pd.get_dummies(df[NUM_FEATURES + CAT_FEATURES], columns=CAT_FEATURES,
                   drop_first=True, dtype=int)
X = X.fillna(X.median())
confounders = list(X.columns)
data = pd.concat([X, df[["is_late", "review_score", "low_review", "delay_days"]]], axis=1)
print(f"Orders with a review: {len(data):,} | late: {data.is_late.mean():.2%} | "
      f"confounders adjusted for: {len(confounders)}")

# ------------------------------------------------------------------ 2. naive vs adjusted effect
rows = []


def naive_diff(outcome):
    g = data.groupby("is_late")[outcome].mean()
    return g[1] - g[0]


def causal_estimate(frame, treatment, outcome, method, **params):
    model = CausalModel(data=frame, treatment=treatment, outcome=outcome,
                        common_causes=confounders)
    # "maximal-adjustment" = adjust for all listed confounders; the default search is very slow with 75 of them
    estimand = model.identify_effect(method_name="maximal-adjustment", proceed_when_unidentifiable=True)
    est = model.estimate_effect(estimand, method_name=method, method_params=params or None)
    return model, estimand, est


models = {}
for outcome, label in [("review_score", "stars (1-5)"), ("low_review", "probability of a 1-2 star review")]:
    naive = naive_diff(outcome)
    m, est_id, est_lr = causal_estimate(data, "is_late", outcome, "backdoor.linear_regression")
    _, _, est_ps = causal_estimate(data, "is_late", outcome, "backdoor.propensity_score_weighting",
                                   weighting_scheme="ips_normalized_weight")
    models[outcome] = (m, est_id, est_lr)
    rows += [
        {"outcome": outcome, "method": "Naive difference (no adjustment)", "effect": naive},
        {"outcome": outcome, "method": "DoWhy: linear regression adjustment", "effect": est_lr.value},
        {"outcome": outcome, "method": "DoWhy: propensity-score weighting", "effect": est_ps.value},
    ]
    print(f"\nEffect of a late delivery on {label}:")
    print(f"  naive difference            : {naive:+.3f}")
    print(f"  regression adjustment       : {est_lr.value:+.3f}")
    print(f"  propensity-score weighting  : {est_ps.value:+.3f}")

results = pd.DataFrame(rows)

# ------------------------------------------------------------------ 3. how bad is "how late"?
# Compare on-time orders with orders that were 1-3, 4-7, 8-14, 15+ days late.
bins = [(0, 3, "1-3 days late"), (3, 7, "4-7 days late"), (7, 14, "8-14 days late"), (14, 1e9, "15+ days late")]
on_time = data[data["delay_days"] <= 0]
dose = []
for lo, hi, name in bins:
    sub = pd.concat([on_time, data[(data["delay_days"] > lo) & (data["delay_days"] <= hi)]]).copy()
    sub["treat"] = (sub["delay_days"] > 0).astype(int)
    _, _, est = causal_estimate(sub, "treat", "review_score", "backdoor.linear_regression")
    n = int(sub["treat"].sum())
    dose.append({"group": name, "orders": n, "stars_change": est.value})
dose = pd.DataFrame(dose)
print("\nEstimated review-score change vs on-time orders, by how late (adjusted):")
print(dose.round(3).to_string(index=False))

# ------------------------------------------------------------------ 4. refutation tests (try to break the result)
print(f"\nRefutation tests on the main estimate (review_score, {N_SIM} simulations each):")
m, est_id, est_lr = models["review_score"]
tests = {
    "Add a random fake confounder (effect should NOT change)":
        dict(method_name="random_common_cause", num_simulations=N_SIM),
    "Shuffle the treatment (effect should drop to ~0)":
        dict(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=N_SIM),
    "Use random 80% subsets (effect should stay similar)":
        dict(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=N_SIM),
}
ref_rows = []
for name, kwargs in tests.items():
    try:
        ref = m.refute_estimate(est_id, est_lr, **kwargs)
        pval = (ref.refutation_result or {}).get("p_value") if isinstance(ref.refutation_result, dict) else None
        ref_rows.append({"test": name, "original_effect": ref.estimated_effect,
                         "new_effect": ref.new_effect, "p_value": pval})
        print(f"  - {name}\n      original {ref.estimated_effect:+.3f} -> new {ref.new_effect:+.3f}"
              + (f"  (p = {pval:.2f})" if pval is not None else ""))
    except Exception as e:                                   # don't let one test stop the script
        print(f"  - {name}: could not run ({type(e).__name__}: {str(e)[:80]})")

# How strong would a hidden confounder need to be?  (simple sensitivity check)
try:
    ref = m.refute_estimate(est_id, est_lr, method_name="add_unobserved_common_cause",
                            confounders_effect_on_treatment="binary_flip",
                            confounders_effect_on_outcome="linear",
                            effect_strength_on_treatment=0.01, effect_strength_on_outcome=0.02)
    print(f"  - Add a hidden confounder (moderate strength)\n      original "
          f"{ref.estimated_effect:+.3f} -> new {ref.new_effect:+.3f}")
    ref_rows.append({"test": "Hidden confounder (moderate)", "original_effect": ref.estimated_effect,
                     "new_effect": ref.new_effect, "p_value": None})
except Exception as e:
    print(f"  - Hidden-confounder check: could not run ({type(e).__name__})")

# ------------------------------------------------------------------ 5. business impact
effect_stars = -results.loc[(results.outcome == "review_score") &
                            results.method.str.contains("regression"), "effect"].iloc[0]
effect_low = -results.loc[(results.outcome == "low_review") &
                          results.method.str.contains("regression"), "effect"].iloc[0]
n_late = int(data.is_late.sum())
print("\n--- Business impact (assumes the causal estimate holds) ---")
print(f"Each late delivery costs about {effect_stars:.2f} review stars "
      f"and raises the chance of a 1-2 star review by {abs(effect_low) * 100:.0f} percentage points.")
print(f"Cutting the late rate by 1 percentage point would lift the average review score by about "
      f"{effect_stars * 0.01:.3f} stars and avoid roughly {int(round(abs(effect_low) * 0.01 * len(data))):,} "
      f"low reviews.")
print(f"Bringing the {n_late:,} late orders in this dataset on time would avoid roughly "
      f"{int(round(abs(effect_low) * n_late)):,} low (1-2 star) reviews.")

# ------------------------------------------------------------------ 6. save tables + charts
results.to_csv(OUT / "causal_estimates.csv", index=False)
dose.to_csv(OUT / "causal_delay_dose_response.csv", index=False)
pd.DataFrame(ref_rows).to_csv(OUT / "causal_refutations.csv", index=False)

stars = results[results.outcome == "review_score"]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(stars["method"], stars["effect"], color=["#9aa0a6", "#1a73e8", "#1a73e8"])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Change in review score caused by a late delivery (stars)")
ax.set_title("Late delivery -> review score")
plt.tight_layout()
plt.savefig(OUT / "causal_effect_estimates.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar(dose["group"], dose["stars_change"], color="#d93025")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Change in review score vs on-time (stars)")
ax.set_title("The later the delivery, the lower the review")
plt.tight_layout()
plt.savefig(OUT / "causal_delay_dose_response.png", dpi=150)
plt.close()

print("\nSaved: outputs/causal_estimates.csv, causal_delay_dose_response.csv, causal_refutations.csv, "
      "causal_effect_estimates.png, causal_delay_dose_response.png")
