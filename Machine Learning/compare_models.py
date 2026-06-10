from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
import pandas as pd
import time

# ==========================================
# PHẦN 1: KHỞI TẠO SPARK
# ==========================================
spark = SparkSession.builder \
    .appName("Compare_Encoding_Methods") \
    .master("local[*]") \
    .getOrCreate()

# ==========================================
# PHẦN 2: THƯỚC ĐO ĐÁNH GIÁ
# ==========================================

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="probability",
    metricName="areaUnderROC"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

# ==========================================
# PHẦN 3: HÀM ĐÁNH GIÁ DATASET
# ==========================================

results = []

def evaluate_dataset(data_path, encoding_name):

    print("\n" + "=" * 60)
    print(f"ENCODING: {encoding_name}")
    print("=" * 60)

    # Đọc dữ liệu
    df = spark.read.parquet(data_path)

    print(f"Số dòng dữ liệu: {df.count()}")

    # Train/Test Split
    train_data, test_data = df.randomSplit(
        [0.7, 0.3],
        seed=42
    )

    print(f"Train: {train_data.count()}")
    print(f"Test : {test_data.count()}")

    # ======================================
    # LOGISTIC REGRESSION
    # ======================================

    print("\n>>> LOGISTIC REGRESSION")

    start_time = time.time()

    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="label"
    )

    lr_model = lr.fit(train_data)

    lr_predictions = lr_model.transform(test_data)

    lr_roc = roc_evaluator.evaluate(lr_predictions)
    lr_f1 = f1_evaluator.evaluate(lr_predictions)

    lr_time = round(time.time() - start_time, 2)

    print(f"ROC-AUC : {lr_roc:.4f}")
    print(f"F1      : {lr_f1:.4f}")
    print(f"Time    : {lr_time}s")

    results.append([
        encoding_name,
        "Logistic Regression",
        round(lr_roc, 4),
        round(lr_f1, 4),
        lr_time
    ])
    # ======================================
    # DECISION TREE
    # ======================================

    print("\n>>> DECISION TREE")

    start_time = time.time()

    dt = DecisionTreeClassifier(
        featuresCol="scaled_features",
        labelCol="label",
        maxDepth=7,
        seed=42
    )

    dt_model = dt.fit(train_data)

    dt_predictions = dt_model.transform(test_data)

    dt_roc = roc_evaluator.evaluate(dt_predictions)
    dt_f1 = f1_evaluator.evaluate(dt_predictions)

    dt_time = round(time.time() - start_time, 2)

    print(f"ROC-AUC : {dt_roc:.4f}")
    print(f"F1      : {dt_f1:.4f}")
    print(f"Time    : {dt_time}s")

    results.append([
        encoding_name,
        "Decision Tree",
        round(dt_roc, 4),
        round(dt_f1, 4),
        dt_time
    ])


    # ======================================
    # RANDOM FOREST
    # ======================================

    print("\n>>> RANDOM FOREST")

    start_time = time.time()

    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="label",
        numTrees=50,
        maxDepth=5,
        seed=42
    )

    rf_model = rf.fit(train_data)

    rf_predictions = rf_model.transform(test_data)

    rf_roc = roc_evaluator.evaluate(rf_predictions)
    rf_f1 = f1_evaluator.evaluate(rf_predictions)

    rf_time = round(time.time() - start_time, 2)

    print(f"ROC-AUC : {rf_roc:.4f}")
    print(f"F1      : {rf_f1:.4f}")
    print(f"Time    : {rf_time}s")

    results.append([
        encoding_name,
        "Random Forest",
        round(rf_roc, 4),
        round(rf_f1, 4),
        rf_time
    ])


# ==========================================
# PHẦN 4: CHẠY DATASET INDEX
# ==========================================

evaluate_dataset(
    "hdfs://localhost:9000/DACK/weather_ml_rain_index",
    "Index Encoding"
)

# ==========================================
# PHẦN 5: CHẠY DATASET OHE
# ==========================================

evaluate_dataset(
    "hdfs://localhost:9000/DACK/weather_ml_rain_ohe",
    "One-Hot Encoding"
)

# ==========================================
# PHẦN 6: TỔNG HỢP KẾT QUẢ
# ==========================================

print("\n")
print("=" * 80)
print("TỔNG HỢP KẾT QUẢ")
print("=" * 80)

results_df = pd.DataFrame(
    results,
    columns=[
        "Encoding",
        "Model",
        "ROC_AUC",
        "F1_Score",
        "Time_Seconds"
    ]
)

print(results_df)

# ==========================================
# PHẦN 7: LƯU CSV
# ==========================================

csv_file = "encoding_comparison_results.csv"

results_df.to_csv(
    csv_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu kết quả:")
print(csv_file)

# ==========================================
# PHẦN 8: TÌM PHƯƠNG ÁN TỐT NHẤT
# ==========================================

best_model = results_df.sort_values(
    by=["ROC_AUC", "F1_Score"],
    ascending=False
).iloc[0]

print("\n")
print("=" * 80)
print("MÔ HÌNH TỐT NHẤT")
print("=" * 80)

print(f"Encoding : {best_model['Encoding']}")
print(f"Model    : {best_model['Model']}")
print(f"ROC-AUC  : {best_model['ROC_AUC']}")
print(f"F1       : {best_model['F1_Score']}")

spark.stop()