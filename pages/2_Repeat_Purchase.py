import pandas as pd
import plotly.express as px
import streamlit as st

from investigation_template import (
    page_header, get_filtered_data, show_empty_state, kpi_row,
    chart_style, evidence_note, pct, add_download, investigation_footer,
)

st.set_page_config(page_title="Repeat Purchase | Insight AI", page_icon="🔁", layout="wide")
page_header(
    "Repeat Purchase Investigation",
    "Customer retention · Purchase frequency · Repeat-customer segments",
    "Which customers return to place another order, and what does repeat behaviour look like in the selected period?",
)
orders, _ = get_filtered_data("repeat_purchase")
if show_empty_state(orders):
    st.stop()

customer_orders = orders.groupby("customer_unique_id").agg(
    orders=("order_id", "nunique"),
    revenue=("items_value", "sum"),
    avg_order_value=("items_value", "mean"),
    last_purchase=("purchase_date", "max"),
).reset_index()
customer_orders["customer_type"] = customer_orders["orders"].map(
    lambda n: "Repeat (2+ orders)" if n > 1 else "One-time"
)
repeat_customers = customer_orders[customer_orders["orders"] > 1]
repeat_rate = pct(len(repeat_customers), len(customer_orders))
repeat_order_share = pct(orders["customer_unique_id"].map(
    customer_orders.set_index("customer_unique_id")["orders"] > 1
).sum(), len(orders))

kpi_row([
    ("Unique customers", f"{len(customer_orders):,}"),
    ("Repeat-customer rate", f"{repeat_rate:.1f}%"),
    ("Orders from repeat customers", f"{repeat_order_share:.1f}%"),
    ("Avg orders / customer", f"{customer_orders['orders'].mean():.2f}"),
])

left, right = st.columns(2)
with left:
    mix = customer_orders["customer_type"].value_counts().rename_axis("customer_type").reset_index(name="customers")
    fig = px.pie(mix, names="customer_type", values="customers", hole=0.55,
                 title="One-time vs repeat customers")
    st.plotly_chart(chart_style(fig), use_container_width=True)
with right:
    frequency = customer_orders["orders"].clip(upper=6).value_counts().sort_index().reset_index()
    frequency.columns = ["orders_per_customer", "customers"]
    frequency["orders_per_customer"] = frequency["orders_per_customer"].astype(str).replace("6", "6+")
    fig = px.bar(frequency, x="orders_per_customer", y="customers",
                 title="Order frequency per customer (6 = 6+)",
                 labels={"orders_per_customer": "Orders in selected period"})
    st.plotly_chart(chart_style(fig), use_container_width=True)

st.subheader("Repeat customers: value comparison")
value = customer_orders.groupby("customer_type").agg(
    customers=("customer_unique_id", "count"),
    revenue_per_customer=("revenue", "mean"),
    avg_orders=("orders", "mean"),
).reset_index()
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(value, x="customer_type", y="revenue_per_customer",
                 title="Average revenue per customer",
                 labels={"customer_type": "", "revenue_per_customer": "Revenue (R$)"})
    st.plotly_chart(chart_style(fig), use_container_width=True)
with c2:
    fig = px.bar(value, x="customer_type", y="avg_orders",
                 title="Average orders per customer",
                 labels={"customer_type": "", "avg_orders": "Orders"})
    st.plotly_chart(chart_style(fig), use_container_width=True)

evidence_note(
    f"{len(repeat_customers):,} of {len(customer_orders):,} customers placed more than one order "
    f"inside the selected date window. The repeat-customer rate is recalculated for this window, "
    f"rather than using a lifetime flag from the full dataset.",
    "The Olist dataset covers a limited historical period. A customer with one order in this window may have purchased before or after it."
)
add_download(customer_orders, "repeat_purchase_customer_summary.csv")
investigation_footer()
