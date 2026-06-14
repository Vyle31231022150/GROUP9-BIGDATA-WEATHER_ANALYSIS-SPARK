from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ==========================================
# KHỞI TẠO SPARK
# ==========================================

spark = SparkSession.builder \
    .appName("ROC_Curve_OHE") \
    .master("local[*]") \
    .getOrCreate()

# ==========================================
# ĐỌC TRAIN / TEST OHE
# ==========================================

train_data = spark.read.parquet(
    "hdfs://master:9000/DACK/weather_ml_rain_ohe_train"
)

test_data = spark.read.parquet(
    "hdfs://master:9000/DACK/weather_ml_rain_ohe_test"
)

print("Train:", train_data.count())
print("Test :", test_data.count())

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="label"
)

lr_model = lr.fit(train_data)

lr_predictions = lr_model.transform(test_data)

# ==========================================
# DECISION TREE
# ==========================================

dt = DecisionTreeClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    maxDepth=7,
    seed=42
)

dt_model = dt.fit(train_data)

dt_predictions = dt_model.transform(test_data)

# ==========================================
# RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    numTrees=50,
    maxDepth=5,
    seed=42
)

rf_model = rf.fit(train_data)

rf_predictions = rf_model.transform(test_data)

# ==========================================
# HÀM TÍNH ROC
# ==========================================

def get_roc_data(predictions):

    pdf = predictions.select(
        "label",
        "probability"
    ).toPandas()

    y_true = pdf["label"]

    y_score = pdf["probability"].apply(
        lambda x: float(x[1])
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    return fpr, tpr, roc_auc

# ==========================================
# TÍNH ROC CHO 3 MODEL
# ==========================================

lr_fpr, lr_tpr, lr_auc = get_roc_data(
    lr_predictions
)

dt_fpr, dt_tpr, dt_auc = get_roc_data(
    dt_predictions
)

rf_fpr, rf_tpr, rf_auc = get_roc_data(
    rf_predictions
)

# ==========================================
# VẼ ROC CURVE
# ==========================================

plt.figure(figsize=(8,6))

plt.plot(
    lr_fpr,
    lr_tpr,
    linewidth=2,
    label=f"Logistic Regression (AUC={lr_auc:.4f})"
)

plt.plot(
    dt_fpr,
    dt_tpr,
    linewidth=2,
    label=f"Decision Tree (AUC={dt_auc:.4f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    linewidth=2,
    label=f"Random Forest (AUC={rf_auc:.4f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - One Hot Encoding"
)

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "roc_curve_ohe.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nĐã lưu:")
print("roc_curve_ohe.png")

spark.stop()