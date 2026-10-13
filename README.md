# Olist E-Commerce Analytics Dashboard

## Setup
```bash
pip install -r requirements.txt
# put the 9 Olist CSVs inside ./data/
streamlit run app.py
```
First run builds `data/processed/orders_master.parquet` and `items_master.parquet` (~30 s), later runs are instant.

## Files
- `data_loader.py` – joins the 9 CSVs into an order-level table (`orders_master`) and item-level table (`items_master`) with engineered features (delivery_days, delay_days, is_late, freight_ratio, same_state, customer_order_count, ...). Reuse `load_data()` in your ML notebook.
- `app.py` – the dashboard (5 tabs: Overview, Products & Sellers, Customers & Geography, Delivery & Reviews, Data & ML Readiness).
