"""
root_cause_tab.py - "Why are deliveries late, and what do we do about it?"

A plain-language page for non-analysts. It reads the results saved by steps 01-03
(outputs/*.csv, models/*.joblib, data/processed/model_table.csv) so nothing heavy is
re-trained when the dashboard opens.

Use it in one of two ways (see the instructions I gave you):
  1) as a new tab in app.py:      import root_cause_tab ;  with tab: root_cause_tab.render()
  2) as its own page (no edits to app.py): pages/1_Root_Cause_and_Fixes.py calling render()

Note: this page always uses the FULL dataset; the sidebar filters of the main
dashboard do not change it (the causal estimates were computed on all orders).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_prep import CAT_FEATURES, NUM_FEATURES

BASE = Path(__file__).resolve().parent
OUT, PROC, MODELS = BASE / "outputs", BASE / "data" / "processed", BASE / "models"
NEEDED = [PROC / "model_table.csv", OUT / "causal_estimates.csv",
          OUT / "causal_delay_dose_response.csv", OUT / "causal_refutations.csv"]

RED, GREY, BLUE = "#d93025", "#9aa0a6", "#1a73e8"
PEAK = 1.5          # a month counts as a "peak" if its late rate is at least 1.5x the average


# ------------------------------------------------------------------ loading helpers
@st.cache_data(show_spinner=False)
def load_table():
    df = pd.read_csv(PROC / "model_table.csv",
                     parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date", "ship_limit"])
    df["handover_late"] = (df["order_delivered_carrier_date"] > df["ship_limit"]).astype(float)
    df.loc[df["order_delivered_carrier_date"].isna(), "handover_late"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_csv(name):
    return pd.read_csv(OUT / name)


@st.cache_data(show_spinner=False)
def adjusted_effect(treat_col, outcome_col, drop=()):
    """Effect of a 0/1 factor on an outcome AFTER comparing like with like
    (same distance, product type, state, season, ...). Same idea as the DoWhy
    regression adjustment used in step 03."""
    df = load_table().dropna(subset=[treat_col, outcome_col])
    X = pd.get_dummies(df[NUM_FEATURES + CAT_FEATURES], columns=CAT_FEATURES, drop_first=True, dtype=float)
    X = X.drop(columns=list(drop), errors="ignore")
    X = X.fillna(X.median())
    A = np.column_stack([np.ones(len(df)), df[treat_col].astype(float).to_numpy(), X.to_numpy()])
    beta = np.linalg.lstsq(A, df[outcome_col].astype(float).to_numpy(), rcond=None)[0]
    return float(beta[1])


@st.cache_data(show_spinner=False)
def model_lift():
    """Share of late orders caught when we only check the riskiest 10/20/30% of orders."""
    try:
        import joblib
        bundle = joblib.load(MODELS / "hgb_late_delivery.joblib")
        X = pd.read_csv(PROC / "X_test.csv")
        y = pd.read_csv(PROC / "y_test.csv").iloc[:, 0].to_numpy()
        order = np.argsort(-bundle["model"].predict_proba(X)[:, 1])
        return {q: float(y[order[: int(len(order) * q)]].sum() / y.sum()) for q in (0.1, 0.2, 0.3)}
    except Exception:
        return None


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def style(fig, height=300, ytitle=None, ypct=False):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)")
    if ytitle:
        fig.update_yaxes(title=ytitle)
    if ypct:
        fig.update_yaxes(tickformat=".0%")
    return fig


# ------------------------------------------------------------------ the page
def render():
    missing = [p.name for p in NEEDED if not p.exists()]
    if missing:
        st.warning("This page needs results from the analysis scripts. Please run "
                   "`01_ml_model.py`, `02_shap_explain.py` and `03_dowhy_causal.py` first. "
                   f"Missing: {', '.join(missing)}")
        return

    df = load_table()
    est = load_csv("causal_estimates.csv")
    dose = load_csv("causal_delay_dose_response.csv")
    refs = load_csv("causal_refutations.csv")

    def e(outcome, key):
        return float(est[(est.outcome == outcome) & est.method.str.contains(key)]["effect"].iloc[0])

    stars_naive, stars_reg, stars_ps = e("review_score", "Naive"), e("review_score", "regression"), e("review_score", "propensity")
    low_naive, low_reg, low_ps = e("low_review", "Naive"), e("low_review", "regression"), e("low_review", "propensity")

    N, late_n = len(df), int(df["is_late"].sum())
    nat = late_n / N
    rv = df.dropna(subset=["review_score"])
    score_ok, score_late = rv.loc[rv.is_late == 0, "review_score"].mean(), rv.loc[rv.is_late == 1, "review_score"].mean()
    low_ok = (rv.loc[rv.is_late == 0, "review_score"] <= 2).mean()
    low_late = (rv.loc[rv.is_late == 1, "review_score"] <= 2).mean()

    # ============================================================== 1. headline
    st.markdown("## Why are deliveries late, and what does it cost us?")
    st.caption(f"Based on {N:,} delivered orders (2016-2018). This page always shows the full dataset; "
               "the sidebar filters do not change it.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Orders delivered late", pct(nat), f"{late_n:,} orders", delta_color="off")
    c2.metric("Average review: on time vs late", f"{score_ok:.2f} to {score_late:.2f} stars")
    c3.metric("Chance of a 1-2 star review", f"{pct(low_ok, 0)} to {pct(low_late, 0)}",
              help="Share of reviews that are 1 or 2 stars: on-time orders vs late orders")

    # ============================================================== 2. naive vs causal
    st.markdown("### Is lateness really the reason customers are unhappy?")
    st.write("A raw comparison can mislead: maybe late orders are also farther away, or are different "
             "products, and *that* is what annoys customers. So we compare two ways.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Naive view: just compare the two groups**")
            st.metric("Late orders score", f"{stars_naive:+.2f} stars", help="Average review of late orders minus on-time orders")
            st.caption("Quick and simple, but it ignores everything else that differs between late and on-time orders.")
    with right:
        with st.container(border=True):
            st.markdown("**Causal view: compare like with like**")
            st.metric("Effect of being late", f"{stars_reg:+.2f} stars",
                      help="After matching for distance, product type, price, state, season and more")
            st.caption("Same distance, product type, price, state and season, so the only real difference is the lateness.")

    view = st.radio("Show the effect on", ["Review score (stars)", "Chance of a 1-2 star review"], horizontal=True)
    if view.startswith("Review"):
        vals, unit = [stars_naive, stars_reg, stars_ps], "stars"
    else:
        vals, unit = [low_naive * 100, low_reg * 100, low_ps * 100], "percentage points"
    fig = go.Figure(go.Bar(y=["Naive (no adjustment)", "Causal: regression", "Causal: weighting"], x=vals,
                           orientation="h", marker_color=[GREY, BLUE, BLUE],
                           text=[f"{v:+.2f}" if unit == "stars" else f"{v:+.0f}" for v in vals], textposition="outside"))
    fig.update_xaxes(title=f"Effect of a late delivery ({unit})", zeroline=True, zerolinecolor="black")
    st.plotly_chart(style(fig, 230))

    close = abs(stars_reg - stars_naive) <= 0.15 * abs(stars_naive)
    if close:
        st.success(f"**Verdict:** the naive and causal answers agree (about {abs(stars_reg):.1f} stars lost). "
                   "So late delivery itself is driving the drop in reviews. It is not just a side effect of "
                   "distance or product mix.")
    else:
        st.warning(f"**Verdict:** the naive answer ({stars_naive:+.2f}) and the causal answer ({stars_reg:+.2f}) "
                   "differ, so other factors are mixed in. Trust the causal figure more.")

    with st.expander("How sure are we? (stress tests)"):
        plain = {
            "random fake": ("Added a random, meaningless factor", "Effect should not change", "same"),
            "shuffle": ("Shuffled which orders were 'late'", "Effect should disappear", "zero"),
            "subset": ("Re-ran on random 80% of the orders", "Effect should stay similar", "same"),
            "hidden": ("Added a hidden factor we cannot see", "Effect may weaken, should not vanish", "weak"),
        }
        rows = []
        for _, r in refs.iterrows():
            key = next((k for k in plain if k in r["test"].lower()), None)
            if key is None:
                continue
            what, expect, kind = plain[key]
            o, n = r["original_effect"], r["new_effect"]
            ok = (abs(n - o) <= 0.05 * abs(o)) if kind == "same" else (abs(n) <= 0.1 * abs(o)) if kind == "zero" \
                else (np.sign(n) == np.sign(o) and abs(n) >= 0.5 * abs(o))
            rows.append({"Stress test": what, "What we hoped to see": expect,
                         "Effect before": f"{o:+.2f}", "Effect after": f"{n:+.2f}",
                         "Result": "Passed" if ok else "Check this"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Passing these tests makes the result more believable. It does not rule out "
                   "factors we have no data on (see limitations below).")

    st.markdown("**The later it arrives, the worse the review**")
    fig = go.Figure(go.Bar(x=dose["group"], y=dose["stars_change"], marker_color=RED,
                           text=[f"{v:+.1f}" for v in dose["stars_change"]], textposition="outside"))
    fig.update_yaxes(title="Change in review score vs on-time (stars)", range=[min(dose["stars_change"]) * 1.2, 0.3])
    st.plotly_chart(style(fig, 300))
    st.caption("Customers forgive a day or two (about half a star), but a week late costs about two stars. "
               "After about 8 days the damage levels off.")

    # ============================================================== 3. where lateness comes from
    st.markdown("### Where do late deliveries come from?")
    a, b = st.columns(2)

    d_h = df.dropna(subset=["handover_late"])
    g = d_h.groupby("handover_late")["is_late"].agg(["mean", "sum", "count"])
    rate_ok, rate_hl = g.loc[0.0, "mean"], g.loc[1.0, "mean"]
    n_hl, lates_hl = int(g.loc[1.0, "count"]), int(g.loc[1.0, "sum"])
    gap_adj = adjusted_effect("handover_late", "is_late", drop=("slack_days",))
    with a:
        st.markdown("**1. Sellers handing over to the carrier late**")
        fig = go.Figure(go.Bar(x=["Handed over on time", "Handed over after deadline"], y=[rate_ok, rate_hl],
                               marker_color=[GREY, RED], text=[pct(rate_ok), pct(rate_hl)], textposition="outside"))
        st.plotly_chart(style(fig, 280, "Orders delivered late", True))
        st.caption(f"Orders handed to the carrier after the seller's deadline are about "
                   f"{rate_hl / rate_ok:.1f}x as likely to arrive late. The gap is {(rate_hl - rate_ok) * 100:.0f} points "
                   f"raw and {gap_adj * 100:.0f} points after comparing like with like, so it is not just distance or "
                   f"product. These {n_hl:,} orders ({pct(n_hl / len(d_h), 0)}) account for {pct(lates_hl / late_n, 0)} of all late deliveries.")

    pb = pd.cut(df["promised_days"], [0, 15, 20, 25, 30, 40, 400],
                labels=["15 days or less", "16-20", "21-25", "26-30", "31-40", "Over 40"])
    pr = df.groupby(pb, observed=True)["is_late"].mean()
    with b:
        st.markdown("**2. Delivery promises that are too tight**")
        fig = go.Figure(go.Bar(x=pr.index.astype(str), y=pr.values, marker_color=[RED] * 2 + [GREY] * 4,
                               text=[pct(v) for v in pr.values], textposition="outside"))
        st.plotly_chart(style(fig, 280, "Orders delivered late", True))
        small = (df.loc[df.is_late == 1, "delay_days"] <= 3).mean()
        st.caption(f"The shorter the promised window, the more often it is missed. "
                   f"{pct(small, 0)} of late orders were late by 3 days or less, so a modest buffer would have kept the promise.")

    c, d = st.columns(2)
    st_tbl = df.groupby("customer_state")["is_late"].agg(["mean", "count"])
    st_tbl = st_tbl[st_tbl["count"] >= 300].sort_values("mean", ascending=False)
    top_states = st_tbl.head(8).iloc[::-1]
    with c:
        st.markdown("**3. Regions with the most late deliveries**")
        fig = go.Figure(go.Bar(y=top_states.index, x=top_states["mean"], orientation="h", marker_color=RED,
                               text=[pct(v) for v in top_states["mean"]], textposition="outside"))
        fig.add_vline(x=nat, line_dash="dash", annotation_text="national average", annotation_position="bottom right")
        fig.update_xaxes(tickformat=".0%", title="Orders delivered late")
        st.plotly_chart(style(fig, 300))
        st.caption("Customer states with at least 300 orders. Northeastern states and Rio de Janeiro stand out.")

    ym = df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["is_late"].agg(["mean", "count"])
    ym = ym[ym["count"] >= 300]
    with d:
        st.markdown("**4. Peak periods**")
        fig = go.Figure(go.Bar(x=[p.strftime("%b %y") for p in ym.index], y=ym["mean"],
                               marker_color=[RED if v >= PEAK * nat else GREY for v in ym["mean"]]))
        fig.add_hline(y=nat, line_dash="dash", annotation_text="average", annotation_position="top left")
        st.plotly_chart(style(fig, 300, "Orders delivered late", True))
        st.caption("Red months were at least 1.5x the average. This covers only two years, so some of it "
                   "may be one-off events rather than a repeating pattern.")

    # ============================================================== 4. what to do
    st.markdown("### What should we do about it?")
    st.write("Each action targets a different cause. Use the slider to see what happens if the fix "
             "works only partly. The impact figures use the causal estimate above.")
    fix = st.slider("How much of each problem could we realistically fix?", 10, 100, 50, 10, format="%d%%") / 100

    pool_hand = n_hl * (rate_hl - rate_ok)
    pool_promise = int(((df.is_late == 1) & (df.delay_days <= 3)).sum())
    hi_states = st_tbl[st_tbl["mean"] > 1.5 * nat]
    pool_region = float((hi_states["count"] * (hi_states["mean"] - nat)).sum())
    hot = ym[(ym["mean"] >= PEAK * nat) & (ym["count"] >= 1000)]
    pool_season = float((hot["count"] * (hot["mean"] - nat)).sum())
    hot_names = ", ".join(p.strftime("%b %Y") for p in hot.index)

    def impact_cols(col, pool, note):
        avoided = pool * fix
        col.metric("Late orders avoided", f"about {avoided:,.0f}")
        col.caption(f"= about {avoided * abs(low_reg):,.0f} fewer 1-2 star reviews and "
                    f"+{avoided * abs(stars_reg) / N:.3f} stars on the average score. {note}")

    cards = [
        ("1. Make seller handover deadlines stick", "High confidence",
         f"**Evidence:** late handovers are linked to {rate_hl / rate_ok:.1f}x more late deliveries, and the gap holds "
         f"after comparing like with like. The problem is widespread: the 10 worst sellers cause only a small share, "
         f"so this needs a rule for everyone, not a few firings.\n\n"
         "**Action:** automatic reminder before each seller's shipping deadline; a daily 'about to miss' list for account "
         "managers; a seller scorecard with consequences for repeated misses.",
         pool_hand, "Targets the extra lateness linked to late handovers."),
        ("2. Promise dates customers can trust", "Medium confidence",
         f"**Evidence:** {pct(pool_promise / late_n, 0)} of late orders missed the promise by 3 days or less, "
         "and tight promises are missed most often.\n\n"
         "**Action:** add a buffer to the estimated delivery date on risky routes (long distance, high-risk states, peak "
         "months). Pilot it on a few routes first and watch sales and reviews.",
         pool_promise, "Upper bound: an honest promise stops it counting as late, but it does not make delivery faster, "
                       "and longer promises could reduce sales. Test before rolling out."),
        ("3. Fix the worst regions", "Medium confidence",
         f"**Evidence:** {len(hi_states)} states ({', '.join(hi_states.index)}) are late at more than 1.5x the national "
         "rate.\n\n"
         "**Action:** review carrier performance on those routes, add a regional carrier or a closer fulfilment point, "
         "and use realistic dates for those customers.",
         pool_region, "Targets the excess lateness in these states above the national average."),
        ("4. Prepare for peak periods", "Lower confidence",
         f"**Evidence:** {hot_names or 'a few months'} were late at least 1.5x as often as average.\n\n"
         "**Action:** find out what happened in those months (volume spikes, carrier capacity). Agree extra carrier "
         "capacity in advance, and message customers early when a delay is likely.",
         pool_season, "Only two years of data, so we cannot tell a repeating season from one-off events."),
    ]
    for title, conf, text, pool, note in cards:
        with st.container(border=True):
            t1, t2 = st.columns([3, 2])
            with t1:
                st.markdown(f"**{title}**  \n:gray[{conf}]")
                st.markdown(text)
            with t2:
                st.caption(f"Late orders in scope: about {pool:,.0f}")
                impact_cols(t2, pool, note)
    st.caption("The four actions overlap (one order can be in several), so please do not add the numbers together.")

    lift = model_lift()
    with st.container(border=True):
        st.markdown("**5. Early warning: spot risky orders before they go wrong**  \n:gray[Supports all of the above]")
        if lift:
            st.write(f"Our prediction model ranks orders by risk. Checking only the riskiest 20% of orders would catch "
                     f"about **{pct(lift[0.2], 0)}** of the late ones (the riskiest 10% catch {pct(lift[0.1], 0)}). "
                     "Use it as a daily watchlist for the operations team to expedite or warn customers, not as an "
                     "automatic decision: roughly 1 in 5 flagged orders really is late.")
        else:
            st.write("Run `01_ml_model.py` to enable this section.")

    sel = d_h[d_h["handover_late"] == 1].groupby("seller_id").agg(late_handovers=("is_late", "size"),
                                                                    late_deliveries=("is_late", "sum"))
    sel = sel.join(df.groupby("seller_id").size().rename("orders"))
    sel = sel[sel["orders"] >= 50].sort_values("late_handovers", ascending=False).head(10)
    sel["Late-handover rate"] = (sel["late_handovers"] / sel["orders"]).map(lambda v: pct(v, 0))
    sel.insert(0, "Seller", [s[:8] + "..." for s in sel.index])
    with st.expander("Sellers to start with (most late handovers, sellers with 50+ orders)"):
        st.dataframe(sel.reset_index(drop=True).rename(columns={"late_handovers": "Late handovers",
                                                               "late_deliveries": "of which arrived late",
                                                               "orders": "Total orders"}), hide_index=True)
        share10 = sel["late_handovers"].sum() / max(int(d_h["handover_late"].sum()), 1)
        st.caption(f"Even together, these 10 sellers cause only {pct(share10, 0)} of all late handovers. "
                   "That is why a rule for everyone matters more than a short blacklist.")

    # ============================================================== 5. limitations
    with st.expander("What this analysis can and cannot tell you", expanded=True):
        st.markdown(
            "- **Strong evidence, not a guarantee.** We compared like with like on everything we have data for. "
            "Things we cannot see (damaged or wrong items, carrier problems, weather, product quality) could still play a part.\n"
            "- **Reviews have other causes too.** We only see the score, not the comment, so we cannot say why each customer was unhappy.\n"
            "- **Limited history.** About two years of data (2016-2018), mostly delivered orders that left a review. "
            "Seasonal patterns may be one-off events.\n"
            "- **The impact numbers are scenarios.** They assume a fix works as described, ignore side effects "
            "(for example longer promises reducing sales) and overlap between actions.\n"
            "- **The risk model is a guide, not a judge.** It ranks orders by risk but is wrong about 4 in 5 flags, "
            "so it should prioritise checks rather than trigger automatic action.\n"
            "- **Best next step:** pilot one or two actions on a few sellers or routes, then measure whether late rates and "
            "reviews move as predicted."
        )
