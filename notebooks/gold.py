# Databricks notebook source

# COMMAND ----------
# CONFIG — update this before running
CATALOG = "your_catalog_name"

# COMMAND ----------

from pyspark.sql.functions import (
    col, sum, count, countDistinct, avg, round, date_trunc,
    min, max, datediff, lit, current_date, dense_rank, when
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F

spark.conf.set("spark.sql.shuffle.partitions", "8")

orders = spark.table(f"{CATALOG}.ecom_silver.silver_orders")
products = spark.table(f"{CATALOG}.ecom_silver.silver_products")
users = spark.table(f"{CATALOG}.ecom_silver.silver_users")

# ── REVENUE BY CATEGORY ───────────────────────────────────────────────────────

gold_revenue_by_category = (
    orders
    .join(products, orders.product_id == products.id)
    .withColumn("order_month", date_trunc("month", col("order_date")))
    .groupBy("category", "order_month")
    .agg(
        round(sum("total_price"), 2).alias("total_revenue"),
        count("order_id").alias("total_orders"),
        countDistinct("user_id").alias("unique_customers"),
        round(avg("total_price"), 2).alias("avg_order_value")
    )
    .orderBy("category", "order_month")
)

gold_revenue_by_category.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.ecom_gold.gold_revenue_by_category")
print(f"gold_revenue_by_category: {gold_revenue_by_category.count()} records")

# ── CUSTOMER SEGMENTS (RFM) ───────────────────────────────────────────────────

order_level = (
    orders
    .filter(col("status") != "cancelled")
    .groupBy("order_id", "user_id", "order_date")
    .agg(round(sum("total_price"), 2).alias("order_value"))
)

rfm = (
    order_level
    .groupBy("user_id")
    .agg(
        datediff(current_date(), max("order_date")).alias("recency_days"),
        countDistinct("order_id").alias("frequency"),
        round(sum("order_value"), 2).alias("monetary")
    )
)

r_window = Window.orderBy("recency_days")
f_window = Window.orderBy(col("frequency").desc())
m_window = Window.orderBy(col("monetary").desc())

rfm_scored = (
    rfm
    .withColumn("r_score", dense_rank().over(r_window))
    .withColumn("f_score", dense_rank().over(f_window))
    .withColumn("m_score", dense_rank().over(m_window))
)

max_rank = rfm_scored.agg(max("r_score")).collect()[0][0]

rfm_final = (
    rfm_scored
    .withColumn("r_score", round((col("r_score") / max_rank) * 5).cast("int"))
    .withColumn("f_score", round((col("f_score") / max_rank) * 5).cast("int"))
    .withColumn("m_score", round((col("m_score") / max_rank) * 5).cast("int"))
    .withColumn("rfm_score", col("r_score") + col("f_score") + col("m_score"))
    .withColumn("segment",
        when(col("rfm_score") >= 13, "Champions")
        .when(col("rfm_score") >= 10, "Loyal Customers")
