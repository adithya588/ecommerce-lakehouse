# Databricks notebook source

# COMMAND ----------
# CONFIG — update this before running
CATALOG = "your_catalog_name"

# COMMAND ----------

from pyspark.sql.functions import col, from_json, explode, sha2, current_timestamp
from pyspark.sql.types import *

spark.conf.set("spark.sql.shuffle.partitions", "8")

# ── PRODUCTS ──────────────────────────────────────────────────────────────────

products_schema = StructType([
    StructField("id", IntegerType()),
    StructField("title", StringType()),
    StructField("category", StringType()),
    StructField("price", DoubleType()),
    StructField("discountPercentage", DoubleType()),
    StructField("rating", DoubleType()),
    StructField("stock", IntegerType()),
    StructField("brand", StringType()),
    StructField("sku", StringType())
])

silver_products = (
    spark.table(f"{CATALOG}.ecom_bronze.bronze_products")
    .select(from_json(col("raw_json"), products_schema).alias("d"))
    .select("d.*")
    .withColumn("ingested_at", current_timestamp())
)

silver_products.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.ecom_silver.silver_products")
print(f"silver_products: {silver_products.count()} records")

# ── USERS ─────────────────────────────────────────────────────────────────────

address_schema = StructType([
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("country", StringType())
])

users_schema = StructType([
    StructField("id", IntegerType()),
    StructField("firstName", StringType()),
    StructField("lastName", StringType()),
    StructField("age", IntegerType()),
    StructField("gender", StringType()),
    StructField("email", StringType()),
    StructField("username", StringType()),
    StructField("address", address_schema)
])

silver_users = (
    spark.table(f"{CATALOG}.ecom_bronze.bronze_users")
    .select(from_json(col("raw_json"), users_schema).alias("d"))
    .select(
        col("d.id"),
        col("d.firstName"),
        col("d.lastName"),
        col("d.age"),
        col("d.gender"),
        sha2(col("d.email"), 256).alias("email_hashed"),
        col("d.username"),
        col("d.address.city").alias("city"),
        col("d.address.state").alias("state"),
        col("d.address.country").alias("country")
    )
    .withColumn("ingested_at", current_timestamp())
)

silver_users.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.ecom_silver.silver_users")
print(f"silver_users: {silver_users.count()} records")

# ── CARTS ─────────────────────────────────────────────────────────────────────

cart_product_schema = StructType([
    StructField("id", IntegerType()),
    StructField("title", StringType()),
    StructField("price", DoubleType()),
    StructField("quantity", IntegerType()),
    StructField("total", DoubleType()),
    StructField("discountPercentage", DoubleType()),
    StructField("discountedTotal", DoubleType())
])

carts_schema = StructType([
    StructField("id", IntegerType()),
    StructField("userId", IntegerType()),
    StructField("total", DoubleType()),
    StructField("discountedTotal", DoubleType()),
    StructField("totalProducts", IntegerType()),
    StructField("totalQuantity", IntegerType()),
    StructField("products", ArrayType(cart_product_schema))
])

silver_carts = (
    spark.table(f"{CATALOG}.ecom_bronze.bronze_carts")
    .select(from_json(col("raw_json"), carts_schema).alias("d"))
    .select(
        col("d.id").alias("cart_id"),
        col("d.userId").alias("user_id"),
        col("d.total").alias("cart_total"),
        col("d.discountedTotal").alias("cart_discounted_total"),
        col("d.totalProducts").alias("total_products"),
        col("d.totalQuantity").alias("total_quantity"),
        explode(col("d.products")).alias("product")
    )
    .select(
        col("cart_id"),
        col("user_id"),
        col("cart_total"),
        col("cart_discounted_total"),
        col("total_products"),
        col("total_quantity"),
        col("product.id").alias("product_id"),
        col("product.title").alias("product_title"),
        col("product.price").alias("unit_price"),
        col("product.quantity").alias("quantity"),
        col("product.total").alias("line_total"),
        col("product.discountPercentage").alias("discount_percentage"),
        col("product.discountedTotal").alias("discounted_line_total")
    )
    .withColumn("ingested_at", current_timestamp())
)

silver_carts.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.ecom_silver.silver_carts")
print(f"silver_carts: {silver_carts.count()} records")

# ── ORDERS ────────────────────────────────────────────────────────────────────

order_item_schema = StructType([
    StructField("product_id", IntegerType()),
    StructField("quantity", IntegerType()),
    StructField("unit_price", DoubleType()),
    StructField("total_price", DoubleType())
])

silver_orders = (
    spark.table(f"{CATALOG}.ecom_bronze.bronze_orders")
    .select(
        col("order_id"),
        col("user_id").cast(IntegerType()),
        col("order_date").cast(TimestampType()),
        col("status"),
        col("order_total").cast(DoubleType()),
        explode(from_json(col("items"), ArrayType(order_item_schema))).alias("item")
    )
    .select(
        col("order_id"),
        col("user_id"),
        col("order_date"),
        col("status"),
        col("order_total"),
        col("item.product_id").alias("product_id"),
        col("item.quantity").alias("quantity"),
        col("item.unit_price").alias("unit_price"),
        col("item.total_price").alias("total_price")
    )
    .withColumn("ingested_at", current_timestamp())
)

silver_orders.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.ecom_silver.silver_orders")
print(f"silver_orders: {silver_orders.count()} records")
