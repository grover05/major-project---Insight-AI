"""
data_prep.py - shared data preparation for Insights AI (Olist dataset).

Builds ONE order-level table used by every later step:
  01_ml_model.py      -> predicts is_late
  02_shap_explain.py  -> explains the saved model
  03_dowhy_causal.py  -> is_late -> review_score (causal question)

Only features known at PURCHASE TIME are used as model inputs, so the model
does not "peek" at the delivery outcome (no data leakage).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

NUM_FEATURES = [
    "n_items", "price", "freight", "freight_ratio", "weight_g", "volume_cm3",
    "distance_km", "promised_days", "same_state","month", "slack_days", "dow", "hour",
]
CAT_FEATURES = ["customer_state", "seller_state", "category"]


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lng points (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_model_table(data_dir=DATA_DIR, top_n_categories=15):
    data_dir = Path(data_dir)
    ts_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv", parse_dates=ts_cols)
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv" , parse_dates=["shipping_limit_date"])
    products = pd.read_csv(data_dir / "olist_products_dataset.csv")
    translation = pd.read_csv(data_dir / "product_category_name_translation.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
    geo = pd.read_csv(data_dir / "olist_geolocation_dataset.csv")
    reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv")

    # ---- target: delivered orders only; late = delivered after the estimated date ----
    orders = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
    ].copy()
    orders["is_late"] = (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    ).astype(int)
    orders["delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    # ---- item-level -> order-level features ----
    products = products.merge(translation, on="product_category_name", how="left")
    items = items.merge(products, on="product_id", how="left")
    items["volume_cm3"] = (
        items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    )
    order_items = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        price=("price", "sum"),
        freight=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"),
        volume_cm3=("volume_cm3", "sum"),
        seller_id=("seller_id", "first"),  # orders with several sellers: first one is used
        ship_limit=("shipping_limit_date", "max"),  # latest handover deadline in the order
        category=("product_category_name_english", "first"),
    ).reset_index()
    order_items["category"] = order_items["category"].fillna("unknown")
    top = order_items["category"].value_counts().head(top_n_categories).index
    order_items["category"] = order_items["category"].where(order_items["category"].isin(top), "other")
    order_items["freight_ratio"] = order_items["freight"] / order_items["price"].replace(0, np.nan)

    # ---- geography: average lat/lng per zip prefix ----
    zip_geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    customers = customers.join(zip_geo, on="customer_zip_code_prefix").rename(
        columns={"geolocation_lat": "c_lat", "geolocation_lng": "c_lng"})
    sellers = sellers.join(zip_geo, on="seller_zip_code_prefix").rename(
        columns={"geolocation_lat": "s_lat", "geolocation_lng": "s_lng"})

    # ---- review score (kept for the DoWhy step, NOT used as an ML input) ----
    review_score = reviews.groupby("order_id")["review_score"].mean().reset_index()

    # ---- combine ----
    df = (
        orders.merge(order_items, on="order_id", how="inner")
        .merge(customers[["customer_id", "customer_unique_id", "customer_state", "c_lat", "c_lng"]],
               on="customer_id", how="left")
        .merge(sellers[["seller_id", "seller_state", "s_lat", "s_lng"]], on="seller_id", how="left")
        .merge(review_score, on="order_id", how="left")
    )

    df["distance_km"] = haversine(df["s_lat"], df["s_lng"], df["c_lat"], df["c_lng"])
    df["promised_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
    df["slack_days"] = (df["ship_limit"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["month"] = df["order_purchase_timestamp"].dt.month
    df["dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["hour"] = df["order_purchase_timestamp"].dt.hour

    return df.reset_index(drop=True)


def get_X_y(df):
    """Model inputs (one-hot encoded) and target. Missing values are left for the caller to fill
    using TRAIN-set medians only."""
    X = pd.get_dummies(df[NUM_FEATURES + CAT_FEATURES], columns=CAT_FEATURES, dtype=int)
    y = df["is_late"]
    return X, y
