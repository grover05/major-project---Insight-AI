import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from investigation_template import (
    page_header, get_filtered_data, show_empty_state, kpi_row,
    chart_style, evidence_note, pct, add_download, investigation_footer,
)

st.set_page_config(page_title="Low Review Drivers | Insight AI", page_icon="⭐", layout="wide")
page_header(
    "Low-Review Drivers",
    "Customer satisfaction · Product, delivery and order characteristics",
    "Which observable order and product characteristics are associated with 1–2 star reviews?",
)
orders, items = get_filtered_data("low_review")
if show_empty_state(orders):
    st.stop()

reviewed = orders[orders["review_score"].notna()].copy()
if reviewed.empty:
    st.warning("No review scores are available for the selected filters.")
    st.stop()
reviewed["review_score"] = reviewed["review_score"].astype(float)
reviewed["low_review"] = reviewed["review_score"] <= 2
low_rate = reviewed["low_review"].mean() * 100

kpi_row([
    ("Reviewed orders", f"{len(reviewed):,}"),
    ("1–2 star reviews", f"{int(reviewed['low_review'].sum()):,}"),
    ("Low-review rate", f"{low_rate:.1f}%"),
    ("Average review", f"{reviewed['review_score'].mean():.2f}/5"),
])

st.subheader("Review score distribution")
dist = reviewed["review_score"].value_counts().sort_index().rename_axis("score").reset_index(name="orders")
fig = px.bar(dist, x="score", y="orders", text="orders",
             title="Number of orders by review score",
             labels={"score": "Review score", "orders": "Orders"})
fig.update_xaxes(dtick=1)
st.plotly_chart(chart_style(fig), use_container_width=True)

st.subheader("Compare low-review rates across potential drivers")
reviewed["delivery_group"] = np.select(
    [reviewed["is_late"] == 1, reviewed["is_late"] == 0],
    ["Late", "On time"], default="Delivery date unavailable"
)
reviewed["freight_band"] = pd.cut(
    reviewed["freight_ratio"].replace([np.inf, -np.inf], np.nan),
    bins=[-np.inf, 0.05, 0.10, 0.20, 0.40, np.inf],
    labels=["0–5%", "5–10%", "10–20%", "20–40%", "40%+"],
)
c1, c2 = st.columns(2)
with c1:
    delivery = reviewed.groupby("delivery_group", observed=True).agg(
        low_review_rate=("low_review", "mean"), orders=("order_id", "nunique")
    ).reset_index()
    delivery["low_review_rate"] *= 100
    fig = px.bar(delivery, x="delivery_group", y="low_review_rate",
                 title="Low-review rate by delivery outcome",
                 labels={"delivery_group": "", "low_review_rate": "1–2 star reviews (%)"})
    st.plotly_chart(chart_style(fig), use_container_width=True)
with c2:
    freight = reviewed.dropna(subset=["freight_band"]).groupby("freight_band", observed=True).agg(
        low_review_rate=("low_review", "mean"), orders=("order_id", "nunique")
    ).reset_index()
    freight["low_review_rate"] *= 100
    fig = px.bar(freight, x="freight_band", y="low_review_rate",
                 title="Low-review rate by freight share",
                 labels={"freight_band": "Freight / item value band", "low_review_rate": "1–2 star reviews (%)"})
    st.plotly_chart(chart_style(fig), use_container_width=True)

if not items.empty and "category" in items.columns:
    st.subheader("Product categories associated with low reviews")
    category = items.merge(
        reviewed[["order_id", "low_review"]].drop_duplicates("order_id"),
        on="order_id", how="inner"
    ).groupby("category").agg(
        items=("order_id", "size"),
        low_review_rate=("low_review", "mean")
    ).reset_index()
    category = category[category["items"] >= 30].nlargest(15, "low_review_rate")
    category["low_review_rate"] *= 100
    fig = px.bar(category.sort_values("low_review_rate"), x="low_review_rate", y="category",
                 orientation="h", title="Categories with highest low-review rate (minimum 30 items)",
                 labels={"low_review_rate": "1–2 star reviews (%)", "category": "Category"})
    st.plotly_chart(chart_style(fig, 430), use_container_width=True)

evidence_note(
    f"The low-review rate in the selected data is {low_rate:.1f}%. Use the segment comparisons to "
    "identify where dissatisfaction is concentrated before investigating operational or product causes.",
    "These are observational associations. Review scores can also reflect product quality, damage, expectations, customer service and unobserved factors."
)
add_download(reviewed, "low_review_filtered_orders.csv")
investigation_footer()
