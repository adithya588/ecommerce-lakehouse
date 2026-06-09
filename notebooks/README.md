# E-Commerce Data Lakehouse on Azure Databricks

An end-to-end data engineering portfolio project built on Azure Databricks demonstrating real-world lakehouse architecture, ML-powered churn prediction, and CI/CD automation.

---




## Data Sources

- **DummyJSON API** — 194 products, 208 users, 208 carts (no auth required)
- **Faker** — 90,000+ synthetic orders referencing real product IDs and prices
- **Auto Loader** — incremental file ingestion from Databricks Volume

---

## Medallion Architecture

### Bronze
Raw ingestion layer — API responses stored as JSON strings, orders ingested via Auto Loader

### Silver
- Parsed and typed using PySpark schemas
- PII masking — email hashed with SHA-256
- Native Delta constraints for data quality

### Gold
| Table | Description |
|---|---|
| `gold_revenue_by_category` | Monthly revenue per product category |
| `gold_customer_segments` | RFM segmentation — Champions, Loyal, At Risk, Lost |
| `gold_product_performance` | Revenue and units sold per product |
| `gold_repeat_vs_new_customers` | Repeat vs new customer split |

---

## ML — Churn Prediction

- **Approach:** Time-windowed behavioral features — 30-day windows per user
- **Model:** XGBoost with `scale_pos_weight` for class imbalance
- **Features:** order frequency, revenue, cancellation rate, category diversity, customer tenure
- **Tracking:** MLflow experiment tracking — params, metrics, model artifacts
- **Registry:** Registered in Unity Catalog Model Registry with `champion` alias
- **Scoring:** Batch predictions written back to Delta table

---

## CI/CD

- **Databricks Asset Bundles** — all jobs defined as code in `databricks.yml`
- **GitHub Actions** — deployment workflow configured to trigger on push to `main` using `DATABRICKS_HOST` and `DATABRICKS_TOKEN` secrets
- **Targets** — `dev` and `prod` environments configured

---

## Setup

1. Clone the repo
2. Update `CATALOG` variable in each notebook to your Unity Catalog name
3. Update `MLFLOW_EXPERIMENT` in `ml.py` to your Databricks email
4. Run `00_setup` SQL commands to create schemas and volume
5. Run notebooks in order: `bronze → silver → gold → ml`
6. Connect Power BI to gold tables via SQL Warehouse

---

## Key Numbers

| Metric | Value |
|---|---|
| Total orders generated | 92,000+ |
| Order line items | 179,000+ |
| ML training samples | 5,000+ (time-windowed) |
| Gold tables | 4 |
| ML model versions tracked | 4 |
