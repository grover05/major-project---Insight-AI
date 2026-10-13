"""
Data loading + feature preparation for the Olist e-commerce dashboard.

Builds two clean tables from the 9 raw CSVs:
  - orders_master : one row per order  (the table your ML model will use later)
  - items_master  : one row per order item (product / seller level analysis)

Processed tables are cached to data/processed/ (parquet) so the app starts fast.
Run `python data_loader.py` to (re)build them manually.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PROC_DIR = DATA_DIR / "processed"
ORDERS_PATH = PROC_DIR / "orders_master.parquet"
ITEMS_PATH = PROC_DIR / "items_master.parquet"


def _read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, **kwargs)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    # ------------------------------------------------------------------ load
    time_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = _read("olist_orders_dataset.csv", parse_dates=time_cols)
    customers = _read("olist_customers_dataset.csv")
    items = _read("olist_order_items_dataset.csv")
    payments = _read("olist_order_payments_dataset.csv")
    reviews = _read("olist_order_reviews_dataset.csv")
    products = _read("olist_products_dataset.csv")
    sellers = _read("olist_sellers_dataset.csv")
    cat_tr = _read("product_category_name_translation.csv")

    # ----------------------------------------------------- item-level table
    products = products.merge(cat_tr, on="product_category_name", how="left")
    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )
    products["volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    items_m = (
        items.merge(
            products[
                [
                    "product_id",
                    "category",
                    "product_weight_g",
                    "volume_cm3",
                    "product_photos_qty",
                ]
            ],
            on="product_id",
            how="left",
        )
        .merge(
            sellers[["seller_id", "seller_city", "seller_state"]],
            on="seller_id",
            how="left",
        )
    )
    items_m["category"] = items_m["category"].fillna("unknown")

    # ----------------------------------------------- order-level aggregates
    items_agg = items_m.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        items_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        avg_photos=("product_photos_qty", "mean"),
    )
    first_item = (
        items_m.sort_values(["order_id", "order_item_id"])
        .drop_duplicates("order_id")[["order_id", "category", "seller_state"]]
        .rename(columns={"category": "main_category"})
    )

    pay_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_type", "nunique"),
    )
    main_pay = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "main_payment_type"})
    )

    reviews["has_comment"] = reviews["review_comment_message"].notna()
    rev_agg = reviews.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        has_review_comment=("has_comment", "max"),
    )
    rev_agg["review_score"] = rev_agg["review_score"].round().astype("Int64")

    # --------------------------------------------------------- master table
    m = (
        orders.merge(
            customers[["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
            on="customer_id",
            how="left",
        )
        .merge(items_agg, on="order_id", how="left")
        .merge(first_item, on="order_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(main_pay, on="order_id", how="left")
        .merge(rev_agg, on="order_id", how="left")
    )

    # time features
    ts = m["order_purchase_timestamp"]
    m["purchase_date"] = ts.dt.normalize()
    m["purchase_month"] = ts.dt.to_period("M").dt.to_timestamp()
    m["purchase_weekday"] = ts.dt.day_name()
    m["purchase_hour"] = ts.dt.hour

    # delivery features (only meaningful when delivered)
    day = 24 * 3600
    m["approval_hours"] = (m["order_approved_at"] - ts).dt.total_seconds() / 3600
    m["delivery_days"] = (m["order_delivered_customer_date"] - ts).dt.total_seconds() / day
    m["estimated_days"] = (m["order_estimated_delivery_date"] - ts).dt.total_seconds() / day
    m["delay_days"] = (
        m["order_delivered_customer_date"] - m["order_estimated_delivery_date"]
    ).dt.total_seconds() / day
    m["is_late"] = np.where(
        m["delay_days"].notna(), (m["delay_days"] > 0).astype(float), np.nan
    )

    # misc features
    m["order_total"] = m["items_value"] + m["freight_value"]
    m["freight_ratio"] = m["freight_value"] / m["items_value"].replace(0, np.nan)
    m["same_state"] = (m["customer_state"] == m["seller_state"]).astype(int)

    cust_orders = m.groupby("customer_unique_id")["order_id"].transform("nunique")
    m["customer_order_count"] = cust_orders
    m["is_repeat_customer"] = (cust_orders > 1).astype(int)

    # item master gets the order-level fields needed for slicing
    items_out = items_m.merge(
        m[["order_id", "order_status", "purchase_month", "customer_state", "review_score",
           "delivery_days", "is_late"]],
        on="order_id",
        how="left",
    )
    return m, items_out


def load_data(force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (orders_master, items_master); builds + caches to parquet on first run."""
    if not force_rebuild and ORDERS_PATH.exists() and ITEMS_PATH.exists():
        return pd.read_parquet(ORDERS_PATH), pd.read_parquet(ITEMS_PATH)

    orders_m, items_m = build_tables()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    orders_m.to_parquet(ORDERS_PATH, index=False)
    items_m.to_parquet(ITEMS_PATH, index=False)
    return orders_m, items_m


if __name__ == "__main__":
    o, i = load_data(force_rebuild=True)
    print(f"orders_master: {o.shape}  ->  {ORDERS_PATH}")
    print(f"items_master : {i.shape}  ->  {ITEMS_PATH}")
