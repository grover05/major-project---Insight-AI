import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from investigation_template import (
    page_header, get_filtered_data, show_empty_state, kpi_row,
    chart_style, evidence_note, money, add_download, investigation_footer,
)

st.set_page_config(page_title="Freight Impact | Insight AI", page_icon="📦", layout="wide")
page_header(
    "Freight Impact Investigation",
    "Shipping cost burden · Product categories · Geography · Customer experience",
    "How large is freight relative to item value, and how does the shipping-cost burden vary across orders and categories?",
)
orders, items = get_filtered_data("freight_impact")
if show_empty_state(orders):
    st.stop()

valid = orders[(orders["items_value"] > 0) & orders["freight_value"].notna()].copy()
if valid.empty:
    st.warning("No orders with valid item value and freight value are available.")
    st.stop()
valid["freight_ratio"] = valid["freight_value"] / valid["items_value"]
valid["freight_share_pct"] = valid["freight_ratio"] * 100
valid["freight_band"] = pd.cut(
    valid["freight_share_pct"],
    bins=[-np.inf, 5, 10, 20, 40, np.inf],
    labels=["0–5%", "5–10%", "10–20%", "20–40%", "40%+"],
)

kpi_row([
    ("Orders analysed", f"{len(valid):,}"),
    ("Total freight", money(valid["freight_value"].sum())),
    ("Average freight / order", money(valid["freight_value"].mean())),
    ("Median freight share", f"{valid['freight_share_pct'].median():.1f}%"),
])

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(valid[valid["freight_share_pct"] < valid["freight_share_pct"].quantile(.99)],
                       x="freight_share_pct", nbins=45,
                       title="Freight as a share of item value (top 1% trimmed)",
                       labels={"freight_share_pct": "Freight / item value (%)"})
    st.plotly_chart(chart_style(fig), use_container_width=True)
with c2:
    band = valid["freight_band"].value_counts().sort_index().rename_axis("freight_band").reset_index(name="orders")
    fig = px.bar(band, x="freight_band", y="orders", title="Orders by freight-burden band",
                 labels={"freight_band": "Freight share", "orders": "Orders"})
    st.plotly_chart(chart_style(fig), use_container_width=True)

st.subheader("Where is freight burden highest?")
if not items.empty:
    category = items.groupby("category").agg(
        item_value=("price", "sum"),
        freight=("freight_value", "sum"),
        items=("order_id", "size"),
    ).reset_index()
    category["freight_share_pct"] = category["freight"] / category["item_value"].replace(0, np.nan) * 100
    category = category[category["items"] >= 30].nlargest(15, "freight_share_pct")
    fig = px.bar(category.sort_values("freight_share_pct"), x="freight_share_pct", y="category",
                 orientation="h", title="Categories with highest freight share (minimum 30 items)",
                 labels={"freight_share_pct": "Freight / item value (%)", "category": "Category"})
    st.plotly_chart(chart_style(fig, 430), use_container_width=True)

state = valid.groupby("customer_state").agg(
    orders=("order_id", "nunique"),
    avg_freight_share=("freight_share_pct", "mean"),
).reset_index()
state = state[state["orders"] >= 20].nlargest(15, "avg_freight_share")
fig = px.bar(state.sort_values("avg_freight_share"), x="avg_freight_share", y="customer_state",
             orientation="h", title="States with highest average freight share (minimum 20 orders)",
             labels={"avg_freight_share": "Average freight share (%)", "customer_state": "Customer state"})
st.plotly_chart(chart_style(fig, 400), use_container_width=True)

if valid["review_score"].notna().any():
    reviewed = valid[valid["review_score"].notna()].copy()
    reviewed["freight_band"] = pd.cut(
        reviewed["freight_share_pct"], bins=[-np.inf, 5, 10, 20, 40, np.inf],
        labels=["0–5%", "5–10%", "10–20%", "20–40%", "40%+"]
    )
    review_impact = reviewed.groupby("freight_band", observed=True).agg(
        avg_review=("review_score", lambda x: x.astype(float).mean()),
        orders=("order_id", "nunique"),
    ).reset_index()
    st.subheader("Freight burden and review scores")
    fig = px.line(review_impact, x="freight_band", y="avg_review", markers=True,
                  title="Average review by freight-burden band",
                  labels={"freight_band": "Freight share", "avg_review": "Average review (1–5)"})
    fig.update_yaxes(range=[0, 5])
    st.plotly_chart(chart_style(fig), use_container_width=True)

evidence_note(
    "Freight share is calculated as freight_value divided by items_value. It is a cost-burden indicator, "
    "not a measure of carrier efficiency or profitability.",
    "Higher freight share may reflect distance, item size/weight, basket value or geography. Do not interpret its association with reviews as causal."
)
add_download(valid, "freight_impact_filtered_orders.csv")
investigation_footer()
