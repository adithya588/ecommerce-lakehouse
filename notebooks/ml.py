# Databricks notebook source

# COMMAND ----------
# CONFIG — update this before running
CATALOG = "your_catalog_name"
MLFLOW_EXPERIMENT = "/Users/your_email@example.com/ecommerce_churn_predictor"

# COMMAND ----------

%pip install xgboost
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from mlflow import MlflowClient
import pyspark.sql.functions as F
from pyspark.sql.window import Window

spark.conf.set("spark.sql.shuffle.partitions", "8")

mlflow.set_registry_uri("databricks-uc")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

orders = spark.table(f"{CATALOG}.ecom_silver.silver_orders")
products = spark.table(f"{CATALOG}.ecom_silver.silver_products")
users = spark.table(f"{CATALOG}.ecom_silver.silver_users")

user_start = (
    orders.filter(F.col("status") != "cancelled")
    .groupBy("user_id")
    .agg(F.min("order_date").alias("first_order_date"))
)

orders_with_window = (
    orders.filter(F.col("status") != "cancelled")
    .join(user_start, "user_id")
    .withColumn("window_num", (F.datediff(F.col("order_date"), F.col("first_order_date")) / 30).cast("int"))
)

window_features = (
    orders_with_window
    .join(products, orders_with_window.product_id == products.id)
    .groupBy("user_id", "window_num", "first_order_date")
    .agg(
        F.count("order_id").alias("orders_in_window"),
        F.round(F.sum("total_price"), 2).alias("revenue_in_window"),
        F.round(F.avg("total_price"), 2).alias("avg_order_value_in_window"),
        F.countDistinct("category").alias("distinct_categories_in_window"),
        F.round(F.sum(F.when(F.col("status") == "cancelled", 1).otherwise(0)) / F.count("order_id"), 4).alias("cancellation_rate_in_window"),
        F.min("order_date").alias("window_start_date")
    )
    .withColumn("days_since_first_ever_order", F.col("window_num") * 30)
)

cum_window = Window.partitionBy("user_id").orderBy("window_num").rowsBetween(Window.unboundedPreceding, Window.currentRow)

window_features_cum = (
    window_features
    .withColumn("cumulative_orders_so_far", F.sum("orders_in_window").over(cum_window))
    .withColumn("cumulative_revenue_so_far", F.round(F.sum("revenue_in_window").over(cum_window), 2))
)

# ── BUILD LABELS ──────────────────────────────────────────────────────────────

user_order_dates = (
    orders.filter(F.col("status") != "cancelled")
    .select("user_id", "order_date")
    .distinct()
)

window_end = window_features_cum.withColumn("window_end_date", F.expr("window_start_date + interval 30 days"))

labeled = (
    window_end.alias("w")
    .join(
        user_order_dates.alias("o"),
        (F.col("w.user_id") == F.col("o.user_id")) &
        (F.col("o.order_date") > F.col("w.window_end_date")) &
        (F.col("o.order_date") <= F.expr("w.window_end_date + interval 30 days")),
        "left"
    )
    .groupBy([F.col("w." + c) for c in window_features_cum.columns])
    .agg(F.count("o.order_date").alias("next_window_orders"))
    .withColumn("churned", F.when(F.col("next_window_orders") == 0, 1).otherwise(0))
    .drop("next_window_orders")
)

labeled_final = labeled.join(users.select(F.col("id").alias("user_id"), "age", "gender"), "user_id")

labeled_final.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{CATALOG}.ecom_ml.ml_features")
print(f"Total rows: {labeled_final.count()}")
labeled_final.groupBy("churned").count().show()

# ── TRAIN MODEL ───────────────────────────────────────────────────────────────

df = spark.table(f"{CATALOG}.ecom_ml.ml_features").toPandas()
df["gender"] = (df["gender"] == "male").astype(int)

feature_cols = [
    "orders_in_window", "revenue_in_window", "avg_order_value_in_window",
    "distinct_categories_in_window", "cancellation_rate_in_window",
    "days_since_first_ever_order", "cumulative_orders_so_far",
    "cumulative_revenue_so_far", "age", "gender"
]

X = df[feature_cols]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scale_pos_weight = int((y == 0).sum() / (y == 1).sum()) if (y == 1).sum() > 0 else 1

# ── MLFLOW ────────────────────────────────────────────────────────────────────

mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name="xgboost_churn_timeseries_v1"):

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_param("min_child_weight", 5)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

    mlflow.xgboost.log_model(
        model,
        artifact_path="churn_model",
        registered_model_name=f"{CATALOG}.ecom_ml.ecommerce_churn_model",
        input_example=X_train[:5]
    )

    print(f"ROC AUC: {auc}")
    print(classification_report(y_test, y_pred))

# ── SET CHAMPION ALIAS ────────────────────────────────────────────────────────

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{CATALOG}.ecom_ml.ecommerce_churn_model'")
latest_version = max([int(v.version) for v in versions])

client.set_registered_model_alias(
    name=f"{CATALOG}.ecom_ml.ecommerce_churn_model",
    alias="champion",
    version=str(latest_version)
)
print(f"Champion alias set to version {latest_version}")

# ── BATCH SCORING ─────────────────────────────────────────────────────────────

latest_features = (
    spark.table(f"{CATALOG}.ecom_ml.ml_features")
    .groupBy("user_id")
    .agg(F.max("window_num").alias("max_window"))
    .join(spark.table(f"{CATALOG}.ecom_ml.ml_features"), "user_id")
    .filter(F.col("window_num") == F.col("max_window"))
    .drop("max_window")
)

score_df = latest_features.toPandas()
score_df["gender"] = (score_df["gender"] == "male").astype(int)

model_uri = f"models:/{CATALOG}.ecom_ml.ecommerce_churn_model@champion"
scoring_model = mlflow.xgboost.load_model(model_uri)

score_df["churn_probability"] = scoring_model.predict_proba(score_df[feature_cols])[:, 1]
score_df["churn_prediction"] = scoring_model.predict(score_df[feature_cols])

predictions_df = spark.createDataFrame(
    score_df[["user_id", "churn_probability", "churn_prediction"]]
).withColumn("scored_at", F.current_timestamp())

predictions_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{CATALOG}.ecom_ml.churn_predictions")
print(f"Scored {predictions_df.count()} users — predictions written to {CATALOG}.ecom_ml.churn_predictions")
