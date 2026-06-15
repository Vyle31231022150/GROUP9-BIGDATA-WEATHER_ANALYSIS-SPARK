import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession

# ==========================================
# KHỞI TẠO SPARK SESSION
# ==========================================

spark = SparkSession.builder \
    .appName("Draw_Comparison_Barchart") \
    .master("local[*]") \
    .getOrCreate()

spark_df = spark.read.csv(
    "hdfs://master:9000/DACK/model_comparison_results",
    header=True,
    inferSchema=True
)

results_df = spark_df.toPandas()

print("Dữ liệu đã đọc thành công:")
print(results_df)

# Dừng Spark sau khi đã lấy xong dữ liệu
spark.stop()

# ==========================================
# CHUẨN BỊ DỮ LIỆU ROC-AUC
# ==========================================

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]

index_auc = []
ohe_auc = []

for model in models:

    index_value = results_df[
        (results_df["Encoding"] == "Index Encoding") &
        (results_df["Model"] == model)
    ]["ROC_AUC"].values[0]

    ohe_value = results_df[
        (results_df["Encoding"] == "One-Hot Encoding") &
        (results_df["Model"] == model)
    ]["ROC_AUC"].values[0]

    index_auc.append(index_value)
    ohe_auc.append(ohe_value)

# ==========================================
# VẼ BIỂU ĐỒ ROC-AUC
# ==========================================

x = np.arange(len(models))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width/2,
    index_auc,
    width,
    label="Index Encoding"
)

plt.bar(
    x + width/2,
    ohe_auc,
    width,
    label="One-Hot Encoding"
)

plt.xticks(x, models)

plt.ylabel("ROC-AUC")

plt.title(
    "ROC-AUC Comparison: Index vs One-Hot Encoding"
)

plt.legend()

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "roc_auc_index_vs_ohe.png",
    dpi=300
)

plt.show()

# ==========================================
# CHUẨN BỊ DỮ LIỆU F1
# ==========================================

index_f1 = []
ohe_f1 = []

for model in models:

    index_value = results_df[
        (results_df["Encoding"] == "Index Encoding") &
        (results_df["Model"] == model)
    ]["F1_Score"].values[0]

    ohe_value = results_df[
        (results_df["Encoding"] == "One-Hot Encoding") &
        (results_df["Model"] == model)
    ]["F1_Score"].values[0]

    index_f1.append(index_value)
    ohe_f1.append(ohe_value)

# ==========================================
# VẼ BIỂU ĐỒ F1
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    x - width/2,
    index_f1,
    width,
    label="Index Encoding"
)

plt.bar(
    x + width/2,
    ohe_f1,
    width,
    label="One-Hot Encoding"
)

plt.xticks(x, models)

plt.ylabel("F1 Score")

plt.title(
    "F1 Score Comparison: Index vs One-Hot Encoding"
)

plt.legend()

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "f1_index_vs_ohe.png",
    dpi=300
)

plt.show()

print("\nĐã lưu:")
print("roc_auc_index_vs_ohe.png")
print("f1_index_vs_ohe.png")