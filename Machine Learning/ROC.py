from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# =====================================================
# SPARK SESSION
# =====================================================

spark = SparkSession.builder \
    .appName("ROC_Curve_Comparison") \
    .master("local[*]") \
    .getOrCreate()

# =====================================================
# LOAD DATASET (OHE)
# =====================================================

data_path = "hdfs://localhost:9000/DACK/weather_ml_rain_ohe"

df = spark.read.parquet(data_path)

train_data, test_data = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Train:", train_data.count())
print("Test :", test_data.count())

# =====================================================
# LOGISTIC REGRESSION
# =====================================================

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="label"
)

lr_model = lr.fit(train_data)

lr_pred = lr_model.transform(test_data)

# =====================================================
# DECISION TREE
# =====================================================

dt = DecisionTreeClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    maxDepth=5
)

dt_model = dt.fit(train_data)

dt_pred = dt_model.transform(test_data)

# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    numTrees=50,
    maxDepth=5
)

rf_model = rf.fit(train_data)

rf_pred = rf_model.transform(test_data)

# =====================================================
# HÀM TRÍCH XÁC SUẤT LỚP 1
# =====================================================

def extract_probability(df_pred):

    pdf = (
        df_pred
        .select("label", "probability")
        .toPandas()
    )

    pdf["prob_1"] = pdf["probability"].apply(
        lambda x: float(x[1])
    )

    return pdf["label"], pdf["prob_1"]

# =====================================================
# LẤY LABEL + PROBABILITY
# =====================================================

y_lr, p_lr = extract_probability(lr_pred)

y_dt, p_dt = extract_probability(dt_pred)

y_rf, p_rf = extract_probability(rf_pred)

# =====================================================
# ROC CURVE
# =====================================================

fpr_lr, tpr_lr, _ = roc_curve(y_lr, p_lr)
roc_lr = auc(fpr_lr, tpr_lr)

fpr_dt, tpr_dt, _ = roc_curve(y_dt, p_dt)
roc_dt = auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = roc_curve(y_rf, p_rf)
roc_rf = auc(fpr_rf, tpr_rf)

# =====================================================
# VẼ BIỂU ĐỒ
# =====================================================

plt.figure(figsize=(10, 7))

plt.plot(
    fpr_lr,
    tpr_lr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {roc_lr:.4f})"
)

plt.plot(
    fpr_dt,
    tpr_dt,
    linewidth=2,
    label=f"Decision Tree (AUC = {roc_dt:.4f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    linewidth=2,
    label=f"Random Forest (AUC = {roc_rf:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison of Machine Learning Models"
)

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "roc_curve_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nROC-AUC")
print("LR :", round(roc_lr, 4))
print("DT :", round(roc_dt, 4))
print("RF :", round(roc_rf, 4))