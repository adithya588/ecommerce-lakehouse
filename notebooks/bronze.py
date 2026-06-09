# Databricks notebook source

# COMMAND ----------
# CONFIG — update this before running
CATALOG = "your_catalog_name"

# COMMAND ----------

%pip install faker
dbutils.library.restartPython()

# COMMAND ----------

import requests
import json
import random
import os
from datetime import datetime, timedelta
from faker import Faker
from pyspark.sql.functions import current_timestamp

fake = Faker()
BASE_URL = "https://dummyjson.com"

# ── API INGESTION ─────────────────────────────────────────────────────────────

def fetch_all_pages(endpoint, key):
    results = []
    skip = 0
    limit = 100
    while True:
        response = requests.get(f"{BASE_URL}/{endpoint}?limit={limit}&skip={skip}")
        data = response.json()
        items = data.get(key, [])
        results.extend(items)
        if skip + limit >= data["total"]:
            break
        skip += limit
    return results

products = fetch_all_pages("products", "products")
users = fetch_all_pages("users", "users")
carts = fetch_all_pages("carts", "carts")

def write_bronze(data, table_name):
    rows = [(json.dumps(record), )]
    df = spark.createDataFrame(rows, ["raw_json"])
    df = df.withColumn("ingested_at", current_timestamp())
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.ecom_bronze.{table_name}")
    print(f"{table_name}: {df.count()} records written")

write_bronze(products, "bronze_products")
write_bronze(users, "bronze_users")
write_bronze(carts, "bronze_carts")

# ── FAKER ORDER GENERATION ────────────────────────────────────────────────────

products_df = spark.sql(f"SELECT raw_json FROM {CATALOG}.ecom_bronze.bronze_products")
products_data = [json.loads(row.raw_json) for row in products_df.collect()]
product_map = {p["id"]: p["price"] for p in products_data}
product_ids = list(product_map.keys())

user_ids = list(range(1, 209))
statuses = ["delivered", "returned", "pending", "cancelled"]
status_weights = [0.7, 0.1, 0.15, 0.05]

end_date = datetime.today()
start_date = end_date - timedelta(days=730)

volume_path = f"/Volumes/{CATALOG}/ecom_bronze/raw_data"
os.makedirs(volume_path, exist_ok=True)

def generate_order():
    order_id = fake.uuid4()
    user_id = random.choice(user_ids)
    order_date = fake.date_time_between(start_date=start_date, end_date=end_date).isoformat()
    status = random.choices(statuses, weights=status_weights)[0]
    num_items = 1 if random.random() < 0.6 else random.randint(2, 5)
    selected_products = random.sample(product_ids, num_items)
    items = []
    for pid in selected_products:
        quantity = random.randint(1, 5)
        price = product_map[pid]
        items.append({
            "product_id": pid,
            "quantity": quantity,
            "unit_price": price,
            "total_price": round(price * quantity, 2)
        })
    return {
        "order_id": order_id,
        "user_id": user_id,
        "order_date": order_date,
        "status": status,
        "items": items,
        "order_total": round(sum(i["total_price"] for i in items), 2)
    }

for file_num in range(1, 91):
    orders = [generate_order() for _ in range(1000)]
    file_path = f"{volume_path}/orders_backfill_{file_num:03d}.json"
    with open(file_path, "w") as f:
        for order in orders:
            f.write(json.dumps(order) + "\n")

print("90 files written — 90,000 orders total")

# ── AUTO LOADER ───────────────────────────────────────────────────────────────

checkpoint_path = f"/Volumes/{CATALOG}/ecom_bronze/raw_data/_checkpoints/bronze_orders"

(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", checkpoint_path)
    .load(f"/Volumes/{CATALOG}/ecom_bronze/raw_data")
    .writeStream
    .format("delta")
    .option("checkpointLocation", checkpoint_path)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .outputMode("append")
    .toTable(f"{CATALOG}.ecom_bronze.bronze_orders")
    .awaitTermination()
)

print("Auto Loader complete — bronze_orders ready")
