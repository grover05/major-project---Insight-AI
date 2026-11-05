"""Reusable investigation utilities for the Insight AI Streamlit pages.

Every investigation page uses the same date/status/state filters, KPI layout,
chart styling, evidence notes and data-loading path.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent


def get_master_data():
    """Load the cached order- and item-level master tables."""
    return load_data()


def get_filtered_data(page_key: str = "investigation"):
    """Shared sidebar filters. Returns filtered orders and matching item rows."""
    orders_all, items_all = get_master_data()
    if orders_all.empty:
        return orders_all, items_all

    st.sidebar.header("Investigation filters")
    min_date = pd.to_datetime(orders_all["purchase_date"]).min().date()
    max_date = pd.to_datetime(orders_all["purchase_date"]).max().date()
    date_range = st.sidebar.date_input(
        "Purchase date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"{page_key}_date",
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date, end_date = min_date, max_date

    statuses = sorted(orders_all["order_status"].dropna().unique().tolist())
    default_status = ["delivered"] if "delivered" in statuses else statuses
    selected_statuses = st.sidebar.multiselect(
        "Order status",
        statuses,
        default=default_status,
        key=f"{page_key}_status",
    )
    states = sorted(orders_all["customer_state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect(
        "Customer state",
        states,
        default=[],
        placeholder="All states",
        key=f"{page_key}_state",
    )

    orders = orders_all.copy()
    orders["purchase_date"] = pd.to_datetime(orders["purchase_date"])
    mask = orders["purchase_date"].dt.date.between(start_date, end_date)
    orders = orders.loc[mask]
    if selected_statuses:
        orders = orders[orders["order_status"].isin(selected_statuses)]
    else:
        orders = orders.iloc[0:0]
    if selected_states:
        orders = orders[orders["customer_state"].isin(selected_states)]

    items = items_all[items_all["order_id"].isin(orders["order_id"])].copy()
    st.sidebar.caption(
        "All charts on this page use the same filters. Repeat-purchase status is "
        "recomputed inside the selected date window."
    )
    return orders.copy(), items


def page_header(title: str, subtitle: str, question: str):
    st.title(title)
    st.caption(subtitle)
    with st.container(border=True):
        st.markdown("**Investigation question**")
        st.write(question)


def kpi_row(metrics):
    """Render metrics as (label, value, optional help) tuples."""
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        label, value = metric[0], metric[1]
        help_text = metric[2] if len(metric) > 2 else None
        col.metric(label, value, help=help_text)


def chart_style(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=55, b=12),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def show_empty_state(df, message="No records match the current filters. Adjust the sidebar filters."):
    if df.empty:
        st.info(message)
        return True
    return False


def evidence_note(text: str, limitation: str = None):
    with st.container(border=True):
        st.markdown("**What the evidence says**")
        st.write(text)
        if limitation:
            st.caption(f"Interpretation caveat: {limitation}")


def pct(numerator, denominator):
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator) * 100


def money(value):
    if pd.isna(value):
        return "—"
    return f"R$ {value:,.2f}"


def add_download(df: pd.DataFrame, filename: str, label="Download filtered data"):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def investigation_footer():
    st.divider()
    st.caption(
        "Insight AI · Investigation workspace · Results are descriptive unless a "
        "causal design is explicitly stated. Association should not be presented as causation."
    )
