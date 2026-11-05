import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from investigation_template import (
    page_header, get_filtered_data, show_empty_state, kpi_row,
    chart_style, evidence_note, money, add_download, investigation_footer,
)

st.set_page_config(page_title="Seller Performance | Insight AI", page_icon="🏪", layout="wide")
page_header(
    "Seller Performance Investigation",
    "Seller scorecard · Fulfilment · Customer reviews · Freight",
    "Which sellers combine meaningful order volume with strong or weak delivery and customer-review outcomes?",
)
orders, items = get_filtered_data("seller_performance")
if show_empty_state(orders):
    st.stop()
if items.empty or "seller_id" not in items.columns:
    st.warning("Seller-level item records are unavailable for the selected filters.")
    st.stop()

order_metrics = orders[[
    "order_id", "delivery_days", "is_late", "review_score", "items_value", "freight_value"
]].drop_duplicates("order_id")
seller = items.groupby("seller_id").agg(
    item_rows=("order_id", "size"),
    orders=("order_id", "nunique"),
    revenue=("price", "sum"),
    freight=("freight_value", "sum"),
    avg_item_price=("price", "mean"),
    seller_state=("seller_state", "first"),
).reset_index()
seller_orders = items[["seller_id", "order_id"]].drop_duplicates().merge(order_metrics, on="order_id", how="left")
seller_outcomes = seller_orders.groupby("seller_id").agg(
    avg_delivery_days=("delivery_days", "mean"),
    late_rate=("is_late", "mean"),
    avg_review=("review_score", lambda x: x.astype(float).mean()),
).reset_index()
seller = seller.merge(seller_outcomes, on="seller_id", how="left")
seller["freight_share_pct"] = seller["freight"] / seller["revenue"].replace(0, np.nan) * 100
seller["late_rate_pct"] = seller["late_rate"] * 100
seller["seller_label"] = seller["seller_id"].astype(str).str[:10] + "…"

min_orders = st.slider("Minimum orders per seller", min_value=1, max_value=50, value=5)
qualified = seller[seller["orders"] >= min_orders].copy()
if qualified.empty:
    st.info("No sellers meet the selected minimum order count.")
    st.stop()

kpi_row([
    ("Sellers meeting threshold", f"{len(qualified):,}"),
    ("Revenue represented", money(qualified["revenue"].sum())),
    ("Weighted late rate", f"{(seller_orders[seller_orders['seller_id'].isin(qualified['seller_id'])]['is_late'].mean() * 100):.1f}%"),
    ("Average seller review", f"{qualified['avg_review'].mean():.2f}/5"),
])

st.subheader("Seller scorecard")
st.caption("Each point is a seller. Bubble size represents order volume; hover for seller details.")
fig = px.scatter(
    qualified, x="late_rate_pct", y="avg_review", size="orders",
    hover_name="seller_label",
    hover_data={"seller_id": True, "orders": True, "revenue": ":,.0f",
                "avg_delivery_days": ":.1f", "freight_share_pct": ":.1f"},
    title="Seller delivery reliability vs customer reviews",
    labels={"late_rate_pct": "Late orders (%)", "avg_review": "Average review (1–5)"},
)
fig.update_yaxes(range=[0, 5])
st.plotly_chart(chart_style(fig, 450), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    top_revenue = qualified.nlargest(12, "revenue").sort_values("revenue")
    fig = px.bar(top_revenue, x="revenue", y="seller_label", orientation="h",
                 title="Top sellers by item revenue",
                 labels={"revenue": "Revenue (R$)", "seller_label": "Seller"})
    st.plotly_chart(chart_style(fig, 400), use_container_width=True)
with c2:
    risk = qualified.nlargest(12, "late_rate_pct").sort_values("late_rate_pct")
    fig = px.bar(risk, x="late_rate_pct", y="seller_label", orientation="h",
                 title="Highest late-delivery rates (qualified sellers)",
                 labels={"late_rate_pct": "Late orders (%)", "seller_label": "Seller"})
    st.plotly_chart(chart_style(fig, 400), use_container_width=True)

st.subheader("Filterable seller table")
display_cols = [
    "seller_id", "seller_state", "orders", "revenue", "freight",
    "freight_share_pct", "avg_delivery_days", "late_rate_pct", "avg_review"
]
st.dataframe(
    qualified[display_cols].sort_values(["orders", "revenue"], ascending=False),
    use_container_width=True,
    hide_index=True,
)
evidence_note(
    "Use the minimum-order threshold to avoid overreacting to sellers with only one or two orders. "
    "Investigate sellers that have both sufficient volume and consistently weak outcome metrics.",
    "Seller comparisons are descriptive. Product mix, geography, order complexity and sample size can differ substantially across sellers."
)
add_download(qualified[display_cols], "seller_performance_scorecard.csv")
investigation_footer()
