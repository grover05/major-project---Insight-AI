import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from investigation_template import (
    page_header, get_filtered_data, show_empty_state, kpi_row,
    chart_style, evidence_note, pct, add_download, investigation_footer,
)

st.set_page_config(page_title="Delivery Investigation | Insight AI", page_icon="🚚", layout="wide")
page_header(
    "Delivery Investigation",
    "Reusable investigation format · Scope → evidence → drivers → actions",
    "Where are delivery failures concentrated, and how are they associated with customer reviews?",
)
orders, _ = get_filtered_data("delivery_investigation")
if show_empty_state(orders):
    st.stop()

delivered = orders[orders["delivery_days"].notna()].copy()
reviewed = delivered[delivered["review_score"].notna()].copy()
if delivered.empty:
    st.warning("No delivered orders are available in this filter selection.")
    st.stop()

late_rate = delivered["is_late"].mean() * 100
avg_days = delivered["delivery_days"].mean()
avg_review = reviewed["review_score"].astype(float).mean() if not reviewed.empty else np.nan
low_review_rate = (reviewed["review_score"].astype(float) <= 2).mean() * 100 if not reviewed.empty else np.nan

kpi_row([
    ("Delivered orders", f"{len(delivered):,}"),
    ("Late-delivery rate", f"{late_rate:.1f}%"),
    ("Average delivery time", f"{avg_days:.1f} days"),
    ("Average review", f"{avg_review:.2f}/5" if pd.notna(avg_review) else "—"),
])

st.subheader("1 · Locate the problem")
left, right = st.columns(2)
with left:
    by_state = delivered.groupby("customer_state").agg(
        orders=("order_id", "nunique"), late_rate=("is_late", "mean")
    ).reset_index()
    by_state = by_state[by_state["orders"] >= 20].sort_values("late_rate", ascending=False).head(15)
    if not by_state.empty:
        by_state["late_rate"] *= 100
        fig = px.bar(by_state, x="late_rate", y="customer_state", orientation="h",
                     title="States with the highest late rate (minimum 20 orders)",
                     labels={"late_rate": "Late orders (%)", "customer_state": "Customer state"})
        st.plotly_chart(chart_style(fig), use_container_width=True)
with right:
    monthly = delivered.groupby("purchase_month").agg(late_rate=("is_late", "mean")).reset_index()
    monthly["late_rate"] *= 100
    fig = px.line(monthly, x="purchase_month", y="late_rate", markers=True,
                  title="Late-delivery rate over purchase months",
                  labels={"purchase_month": "Purchase month", "late_rate": "Late orders (%)"})
    st.plotly_chart(chart_style(fig), use_container_width=True)

st.subheader("2 · Connect delivery outcomes to customer experience")
if not reviewed.empty:
    reviewed["delivery_group"] = np.where(reviewed["is_late"] == 1, "Late", "On time")
    comparison = reviewed.groupby("delivery_group").agg(
        avg_review=("review_score", lambda x: x.astype(float).mean()),
        low_review_rate=("review_score", lambda x: (x.astype(float) <= 2).mean() * 100),
        orders=("order_id", "nunique"),
    ).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(comparison, x="delivery_group", y="avg_review", text="avg_review",
                     title="Average review by delivery outcome",
                     labels={"delivery_group": "", "avg_review": "Average review (1–5)"})
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(chart_style(fig), use_container_width=True)
    with c2:
        fig = px.bar(comparison, x="delivery_group", y="low_review_rate", text="low_review_rate",
                     title="1–2 star review rate by delivery outcome",
                     labels={"delivery_group": "", "low_review_rate": "Low reviews (%)"})
        st.plotly_chart(chart_style(fig), use_container_width=True)
    on_time = comparison.loc[comparison.delivery_group == "On time", "avg_review"]
    late = comparison.loc[comparison.delivery_group == "Late", "avg_review"]
    if len(on_time) and len(late):
        gap = float(on_time.iloc[0] - late.iloc[0])
        evidence_note(
            f"Within the selected filters, late orders have an average review gap of "
            f"{gap:.2f} stars compared with on-time orders.",
            "This is a filtered descriptive comparison, not a causal estimate. Use the separate DoWhy analysis for adjusted causal estimates."
        )
else:
    st.info("There are no reviewed delivered orders in this filter selection.")

st.subheader("3 · Investigate operational drivers")
drivers = delivered.copy()
drivers["route_type"] = np.where(
    drivers["same_state"].fillna(0).astype(int) == 1,
    "Seller and customer in same state",
    "Cross-state shipment",
)
driver_summary = drivers.groupby("route_type").agg(
    orders=("order_id", "nunique"), late_rate=("is_late", "mean")
).reset_index()
driver_summary["late_rate"] *= 100
fig = px.bar(driver_summary, x="route_type", y="late_rate",
             title="Late rate by shipping route type",
             labels={"route_type": "Route type", "late_rate": "Late orders (%)"})
st.plotly_chart(chart_style(fig), use_container_width=True)

evidence_note(
    "Use the state and month charts to identify where to investigate first, then compare same-state and cross-state routes. "
    "Prioritise segments with both meaningful order volume and elevated late rates.",
    "Route type is a broad geography proxy; it does not identify carrier-level causes."
)
add_download(delivered, "delivery_investigation_filtered_orders.csv")
investigation_footer()
