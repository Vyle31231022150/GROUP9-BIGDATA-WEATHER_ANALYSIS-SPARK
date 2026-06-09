from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# ==========================================
# KHỞI TẠO SPARK
# ==========================================
spark = SparkSession.builder \
    .appName("DecisionTree_Debug") \
    .master("local[*]") \
    .getOrCreate()

# ==========================================
# ĐỌC DỮ LIỆU
# ==========================================
data_path = "hdfs://localhost:9000/DACK/weather_ml_rain_ohe"

df = spark.read.parquet(data_path)

print("=" * 60)
print("THÔNG TIN DATASET")
print("=" * 60)

print("Tổng số dòng:", df.count())

df.groupBy("label").count().show()

# ==========================================
# CHIA TRAIN TEST
# ==========================================
train_data, test_data = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Train:", train_data.count())
print("Test :", test_data.count())

# ==========================================
# EVALUATOR
# ==========================================
roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

# ==========================================
# THỬ NHIỀU DEPTH
# ==========================================
depths = [3, 5, 7, 10, 15]

print("\n")
print("=" * 80)
print("KIỂM TRA UNDERFIT / OVERFIT")
print("=" * 80)

for depth in depths:

    print("\n")
    print("-" * 60)
    print(f"MAX DEPTH = {depth}")
    print("-" * 60)

    dt = DecisionTreeClassifier(
        featuresCol="scaled_features",
        labelCol="label",
        maxDepth=depth,
        seed=42
    )

    model = dt.fit(train_data)

    print(f"Depth thực tế : {model.depth}")
    print(f"Số node       : {model.numNodes}")

    # =========================
    # TRAIN
    # =========================
    train_pred = model.transform(train_data)

    train_roc = roc_evaluator.evaluate(train_pred)
    train_f1 = f1_evaluator.evaluate(train_pred)

    # =========================
    # TEST
    # =========================
    test_pred = model.transform(test_data)

    test_roc = roc_evaluator.evaluate(test_pred)
    test_f1 = f1_evaluator.evaluate(test_pred)

    print("\nTRAIN")
    print(f"ROC-AUC : {train_roc:.4f}")
    print(f"F1      : {train_f1:.4f}")

    print("\nTEST")
    print(f"ROC-AUC : {test_roc:.4f}")
    print(f"F1      : {test_f1:.4f}")

    print("\nGAP")
    print(
        f"ROC GAP : {abs(train_roc - test_roc):.4f}"
    )
    print(
        f"F1 GAP  : {abs(train_f1 - test_f1):.4f}"
    )

# ==========================================
# KIỂM TRA PROBABILITY
# ==========================================
print("\n")
print("=" * 80)
print("KIỂM TRA PREDICTION")
print("=" * 80)

dt = DecisionTreeClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    maxDepth=10,
    seed=42
)

model = dt.fit(train_data)

pred = model.transform(test_data)

pred.select(
    "label",
    "prediction",
    "rawPrediction",
    "probability"
).show(20, truncate=False)

# ==========================================
# XUẤT CSV ĐỂ VẼ ROC CURVE
# ==========================================
pred.select(
    "label",
    "probability"
).toPandas().to_csv(
    "decision_tree_probability.csv",
    index=False
)

print("\nĐã xuất file:")
print("decision_tree_probability.csv")