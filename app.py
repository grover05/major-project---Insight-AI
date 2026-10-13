"""
Olist Brazilian E-Commerce — Analytics Dashboard
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_data

st.set_page_config(page_title="Olist E-Commerce Analytics", page_icon="🛒", layout="wide")

# Streamlit >=1.50 uses width="stretch"; older versions use use_container_width
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
FULL = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

PRIMARY = "#2E5BFF"
ACCENT = "#FF6B35"
GOOD = "#1FA97A"
BAD = "#E5484D"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing data (first run only)...")
def get_data():
    return load_data()


orders_all, items_all = get_data()


def fmt_money(x: float) -> str:
    if x >= 1e6:
        return f"R$ {x / 1e6:,.2f}M"
    if x >= 1e3:
        return f"R$ {x / 1e3:,.1f}K"
    return f"R$ {x:,.0f}"


def style(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
        title_font_size=16,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.title("🛒 Filters")

min_d = orders_all["purchase_date"].min().date()
max_d = orders_all["purchase_date"].max().date()
date_range = st.sidebar.date_input(
    "Purchase date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = min_d, max_d

all_status = sorted(orders_all["order_status"].dropna().unique())
status_sel = st.sidebar.multiselect(
    "Order status", all_status, default=["delivered"],
    help="Default = delivered only, so revenue and delivery KPIs are not distorted by cancelled orders.",
)

all_states = sorted(orders_all["customer_state"].dropna().unique())
state_sel = st.sidebar.multiselect("Customer state", all_states, default=[], placeholder="All states")

st.sidebar.caption(
    "Note: data is sparse in late 2016 and after Aug 2018. "
    "Use the date filter to focus on Jan 2017 – Aug 2018 for clean trends."
)

# base = date + state filter (status chart needs all statuses); df = + status filter
base = orders_all[
    (orders_all["purchase_date"].dt.date >= start) & (orders_all["purchase_date"].dt.date <= end)
]
if state_sel:
    base = base[base["customer_state"].isin(state_sel)]
df = base[base["order_status"].isin(status_sel)] if status_sel else base.iloc[0:0]

if df.empty:
    st.warning("No orders match the current filters. Adjust the filters in the sidebar.")
    st.stop()

items = items_all[items_all["order_id"].isin(df["order_id"])]

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.title("Olist E-Commerce Analytics")
st.caption(f"{len(df):,} orders  |  {start:%d %b %Y} → {end:%d %b %Y}")

revenue = df["items_value"].sum()
n_orders = len(df)
n_customers = df["customer_unique_id"].nunique()
aov = df["items_value"].mean()
avg_review = df["review_score"].astype("float").mean()
delivered = df[df["delivery_days"].notna()]
avg_delivery = delivered["delivery_days"].mean() if len(delivered) else np.nan
late_pct = delivered["is_late"].mean() * 100 if len(delivered) else np.nan
repeat_pct = (
    df.groupby("customer_unique_id")["order_id"].nunique().gt(1).mean() * 100
)

r1 = st.columns(4)
r1[0].metric("Total Orders", f"{n_orders:,}")
r1[1].metric("Revenue (items)", fmt_money(revenue))
r1[2].metric("Avg Order Value", f"R$ {aov:,.2f}")
r1[3].metric("Unique Customers", f"{n_customers:,}")
r2 = st.columns(4)
r2[0].metric("Avg Review Score", f"{avg_review:.2f} / 5")
r2[1].metric("Avg Delivery Time", f"{avg_delivery:.1f} days")
r2[2].metric("Late Deliveries", f"{late_pct:.1f}%", help="Delivered after the estimated date")
r2[3].metric("Repeat Customers", f"{repeat_pct:.1f}%", help="Customers with more than one order (within filters)")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "📦 Products & Sellers", "🌎 Customers & Geography", "🚚 Delivery & Reviews", "🧪 Data & ML Readiness"]
)

# ----------------------------------------------------------------------------
# Tab 1: Overview
# ----------------------------------------------------------------------------
with tab1:
    monthly = (
        df.groupby("purchase_month")
        .agg(revenue=("items_value", "sum"), orders=("order_id", "count"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_bar(x=monthly["purchase_month"], y=monthly["revenue"], name="Revenue (R$)", marker_color=PRIMARY, opacity=0.85)
    fig.add_scatter(x=monthly["purchase_month"], y=monthly["orders"], name="Orders", yaxis="y2",
                    mode="lines+markers", line=dict(color=ACCENT, width=3))
    fig.update_layout(
        title="Monthly Revenue & Orders",
        yaxis=dict(title="Revenue (R$)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(style(fig, 400), **FULL)

    c1, c2 = st.columns(2)
    with c1:
        status_df = base["order_status"].value_counts().reset_index()
        status_df.columns = ["status", "orders"]
        fig = px.bar(status_df, x="orders", y="status", orientation="h", text="orders",
                     title="Order Status (ignores status filter)", color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig), **FULL)
    with c2:
        pay = df.groupby("main_payment_type")["payment_value"].sum().reset_index()
        fig = px.pie(pay, names="main_payment_type", values="payment_value", hole=0.5,
                     title="Payment Method Share (by value)")
        st.plotly_chart(style(fig), **FULL)

    c3, c4 = st.columns(2)
    with c3:
        inst = df["max_installments"].clip(upper=12).value_counts().sort_index().reset_index()
        inst.columns = ["installments", "orders"]
        fig = px.bar(inst, x="installments", y="orders", title="Installments per Order (12 = 12+)",
                     color_discrete_sequence=[ACCENT])
        fig.update_xaxes(dtick=1)
        st.plotly_chart(style(fig), **FULL)
    with c4:
        fig = px.histogram(df[df["items_value"] < df["items_value"].quantile(0.99)], x="items_value", nbins=50,
                           title="Order Value Distribution (top 1% trimmed)", color_discrete_sequence=[PRIMARY])
        fig.update_xaxes(title="Order value (R$)")
        st.plotly_chart(style(fig), **FULL)

# ----------------------------------------------------------------------------
# Tab 2: Products & Sellers
# ----------------------------------------------------------------------------
with tab2:
    cat = (
        items.groupby("category")
        .agg(revenue=("price", "sum"), items_sold=("order_id", "count"),
             avg_price=("price", "mean"), avg_freight=("freight_value", "mean"),
             avg_review=("review_score", lambda s: s.astype("float").mean()))
        .reset_index()
    )
    cat["freight_pct_of_price"] = cat["avg_freight"] / cat["avg_price"] * 100

    c1, c2 = st.columns(2)
    with c1:
        top = cat.nlargest(10, "revenue")
        fig = px.bar(top, x="revenue", y="category", orientation="h", title="Top 10 Categories by Revenue",
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig, 420), **FULL)
    with c2:
        top = cat.nlargest(10, "items_sold")
        fig = px.bar(top, x="items_sold", y="category", orientation="h", title="Top 10 Categories by Items Sold",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig, 420), **FULL)

    sub = cat[cat["items_sold"] >= 100]
    fig = px.scatter(sub, x="avg_price", y="avg_review", size="items_sold", hover_name="category",
                     color="freight_pct_of_price", color_continuous_scale="RdYlGn_r",
                     title="Category Map: Price vs Satisfaction (bubble = items sold, colour = freight % of price)",
                     labels={"avg_price": "Avg price (R$)", "avg_review": "Avg review score",
                             "freight_pct_of_price": "Freight % of price"})
    st.plotly_chart(style(fig, 450), **FULL)

    c3, c4 = st.columns(2)
    with c3:
        sellers = items.groupby("seller_id")["price"].sum().sort_values(ascending=False)
        top10 = sellers.head(10).reset_index()
        top10["seller"] = top10["seller_id"].str[:8] + "…"
        fig = px.bar(top10, x="price", y="seller", orientation="h", title="Top 10 Sellers by Revenue",
                     color_discrete_sequence=[PRIMARY], labels={"price": "Revenue (R$)"})
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig), **FULL)
    with c4:
        cum = sellers.cumsum() / sellers.sum() * 100
        pareto = pd.DataFrame({"pct_sellers": np.arange(1, len(cum) + 1) / len(cum) * 100, "pct_revenue": cum.values})
        fig = px.line(pareto, x="pct_sellers", y="pct_revenue", title="Seller Concentration (Pareto curve)",
                      labels={"pct_sellers": "% of sellers", "pct_revenue": "% of revenue"})
        fig.update_traces(line=dict(color=ACCENT, width=3))
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="grey"))
        st.plotly_chart(style(fig), **FULL)
        top20_share = pareto.loc[(pareto["pct_sellers"] - 20).abs().idxmin(), "pct_revenue"]
        st.caption(f"Top 20% of sellers generate **{top20_share:.0f}%** of revenue ({len(sellers):,} active sellers).")

    top_freight = cat[cat["items_sold"] >= 100].nlargest(10, "freight_pct_of_price")
    fig = px.bar(top_freight, x="freight_pct_of_price", y="category", orientation="h",
                 title="Categories where Freight is Heaviest (freight as % of price)",
                 color_discrete_sequence=[BAD], labels={"freight_pct_of_price": "Freight % of price"})
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(style(fig), **FULL)

# ----------------------------------------------------------------------------
# Tab 3: Customers & Geography
# ----------------------------------------------------------------------------
with tab3:
    state = (
        df.groupby("customer_state")
        .agg(revenue=("items_value", "sum"), orders=("order_id", "count"),
             avg_delivery=("delivery_days", "mean"), late_pct=("is_late", lambda s: s.mean() * 100),
             avg_review=("review_score", lambda s: s.astype("float").mean()))
        .reset_index()
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(state.sort_values("revenue", ascending=False).head(15), x="customer_state", y="revenue",
                     title="Revenue by Customer State (top 15)", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(style(fig), **FULL)
    with c2:
        city = df.groupby("customer_city")["items_value"].sum().nlargest(10).reset_index()
        fig = px.bar(city, x="items_value", y="customer_city", orientation="h", title="Top 10 Cities by Revenue",
                     color_discrete_sequence=[ACCENT], labels={"items_value": "Revenue (R$)", "customer_city": "City"})
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig), **FULL)

    heat = df.groupby(["purchase_weekday", "purchase_hour"]).size().reset_index(name="orders")
    heat = heat.pivot(index="purchase_weekday", columns="purchase_hour", values="orders").reindex(WEEKDAYS)
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(x="Hour of day", y="", color="Orders"), title="When do customers buy? (weekday × hour)")
    st.plotly_chart(style(fig, 360), **FULL)

    c3, c4 = st.columns(2)
    with c3:
        cust = df.groupby("customer_unique_id")["order_id"].nunique().clip(upper=4).value_counts().sort_index().reset_index()
        cust.columns = ["orders_per_customer", "customers"]
        cust["orders_per_customer"] = cust["orders_per_customer"].astype(str).replace("4", "4+")
        fig = px.bar(cust, x="orders_per_customer", y="customers", text="customers",
                     title="Orders per Customer (most buy only once)", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(style(fig), **FULL)
    with c4:
        s = state[state["orders"] >= 30]
        fig = px.scatter(s, x="avg_delivery", y="avg_review", size="orders", hover_name="customer_state",
                         title="State Map: Delivery Time vs Review Score (bubble = orders)",
                         labels={"avg_delivery": "Avg delivery days", "avg_review": "Avg review score"},
                         color_discrete_sequence=[ACCENT])
        st.plotly_chart(style(fig), **FULL)

# ----------------------------------------------------------------------------
# Tab 4: Delivery & Reviews
# ----------------------------------------------------------------------------
with tab4:
    rv = df[df["review_score"].notna()].copy()
    rv["review_score"] = rv["review_score"].astype(int)

    c1, c2 = st.columns(2)
    with c1:
        dist = rv["review_score"].value_counts().sort_index().reset_index()
        dist.columns = ["score", "orders"]
        dist["pct"] = dist["orders"] / dist["orders"].sum() * 100
        fig = px.bar(dist, x="score", y="orders", text=dist["pct"].map("{:.0f}%".format),
                     title="Review Score Distribution", color="score",
                     color_continuous_scale=[BAD, "#F5A524", GOOD])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig), **FULL)
    with c2:
        d = delivered[delivered["delivery_days"] < 60]
        fig = go.Figure()
        fig.add_histogram(x=d["delivery_days"], name="Actual delivery", marker_color=PRIMARY, opacity=0.75, nbinsx=60)
        fig.add_histogram(x=d["estimated_days"], name="Estimated delivery", marker_color=ACCENT, opacity=0.6, nbinsx=60)
        fig.update_layout(barmode="overlay", title="Actual vs Estimated Delivery Time (days)")
        st.plotly_chart(style(fig), **FULL)

    # delay bucket -> review score (the key relationship)
    dl = delivered[delivered["review_score"].notna()].copy()
    dl["bucket"] = pd.cut(
        dl["delay_days"], bins=[-np.inf, -7, 0, 3, 7, np.inf],
        labels=["7+ days early", "On time (0-7 early)", "1-3 days late", "4-7 days late", "7+ days late"],
    )
    bk = dl.groupby("bucket", observed=True).agg(
        avg_review=("review_score", lambda s: s.astype("float").mean()), orders=("order_id", "count")
    ).reset_index()
    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(bk, x="bucket", y="avg_review", text=bk["avg_review"].round(2), color="avg_review",
                     color_continuous_scale=[BAD, "#F5A524", GOOD], range_color=[1, 5],
                     title="Avg Review Score by Delivery Delay", labels={"bucket": "", "avg_review": "Avg review"})
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 5])
        st.plotly_chart(style(fig), **FULL)
    with c4:
        late_state = state[state["orders"] >= 30].nlargest(12, "late_pct")
        fig = px.bar(late_state, x="late_pct", y="customer_state", orientation="h",
                     title="States with Highest Late-Delivery Rate", color_discrete_sequence=[BAD],
                     labels={"late_pct": "% late", "customer_state": "State"})
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig), **FULL)

    c5, c6 = st.columns(2)
    with c5:
        trend = delivered.groupby("purchase_month").agg(late_pct=("is_late", lambda s: s.mean() * 100)).reset_index()
        fig = px.line(trend, x="purchase_month", y="late_pct", markers=True, title="Late-Delivery Rate over Time (%)")
        fig.update_traces(line=dict(color=BAD, width=3))
        st.plotly_chart(style(fig), **FULL)
    with c6:
        ss = dl.assign(route=np.where(dl["same_state"] == 1, "Seller & customer same state", "Different states"))
        route = ss.groupby("route").agg(
            avg_delivery=("delivery_days", "mean"),
            avg_review=("review_score", lambda s: s.astype("float").mean()),
        ).reset_index()
        fig = px.bar(route, x="route", y="avg_delivery", text=route["avg_delivery"].round(1),
                     title="Avg Delivery Days: Same-State vs Cross-State", color_discrete_sequence=[PRIMARY],
                     labels={"route": "", "avg_delivery": "Avg delivery days"})
        st.plotly_chart(style(fig), **FULL)

    if len(dl) and dl["is_late"].nunique() > 1:
        late_r = dl.loc[dl["is_late"] == 1, "review_score"].astype(float).mean()
        ont_r = dl.loc[dl["is_late"] == 0, "review_score"].astype(float).mean()
        st.info(f"**Insight:** late orders average a review score of **{late_r:.2f}** vs **{ont_r:.2f}** for on-time orders "
                f"(gap of {ont_r - late_r:.2f} points).")

# ----------------------------------------------------------------------------
# Tab 5: Data & ML readiness
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Order-level master table")
    st.caption("One row per order, joined from all 8 source tables. This is the table to build your ML dataset from.")
    a, b, c = st.columns(3)
    a.metric("Rows (filtered)", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]}")
    c.metric("Date span", f"{(df['purchase_date'].max() - df['purchase_date'].min()).days} days")
    st.dataframe(df.head(200), **FULL, height=300)

    c1, c2 = st.columns(2)
    with c1:
        miss = (df.isna().mean() * 100).round(2)
        miss = miss[miss > 0].sort_values(ascending=False).reset_index()
        miss.columns = ["column", "% missing"]
        fig = px.bar(miss, x="% missing", y="column", orientation="h", title="Missing Values by Column",
                     color_discrete_sequence=[BAD])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig, max(300, 28 * len(miss) + 80)), **FULL)
    with c2:
        num_cols = ["items_value", "freight_value", "freight_ratio", "n_items", "n_sellers", "total_weight_g",
                    "total_volume_cm3", "max_installments", "payment_value", "approval_hours", "delivery_days",
                    "estimated_days", "delay_days", "is_late", "review_score", "same_state", "customer_order_count"]
        corr = df[num_cols].astype(float).corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation Matrix (numeric features)")
        fig.update_traces(textfont_size=8)
        st.plotly_chart(style(fig, 560), **FULL)

    st.subheader("Class balance of candidate targets")
    t1, t2, t3 = st.columns(3)
    lt = delivered["is_late"].value_counts(normalize=True).rename({0.0: "On time", 1.0: "Late"}) * 100
    t1.plotly_chart(style(px.bar(lt, title="is_late (%)", color_discrete_sequence=[BAD]).update_layout(showlegend=False, xaxis_title="", yaxis_title="%"), 300), **FULL)
    rs = rv["review_score"].apply(lambda x: "Low (1-3)" if x <= 3 else "High (4-5)").value_counts(normalize=True) * 100
    t2.plotly_chart(style(px.bar(rs, title="Low vs high review (%)", color_discrete_sequence=[PRIMARY]).update_layout(showlegend=False, xaxis_title="", yaxis_title="%"), 300), **FULL)
    rp = df["customer_order_count"].gt(1).value_counts(normalize=True).rename({False: "One-time", True: "Repeat"}) * 100
    t3.plotly_chart(style(px.bar(rp, title="Repeat customer (%)", color_discrete_sequence=[ACCENT]).update_layout(showlegend=False, xaxis_title="", yaxis_title="%"), 300), **FULL)

    st.download_button(
        "⬇️ Download filtered order-level table (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="orders_master_filtered.csv",
        mime="text/csv",
    )
