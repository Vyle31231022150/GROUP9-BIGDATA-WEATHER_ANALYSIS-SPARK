from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

import pandas as pd
import time

# ==========================================
# PHẦN 1: KHỞI TẠO SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("Models")
    .master("spark://26.49.14.99:7077")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "2g")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
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

results = []

# ==========================================
# PHẦN 3: HÀM ĐÁNH GIÁ DATASET
# ==========================================

def evaluate_dataset(
        train_path,
        test_path,
        encoding_name):

    print("\n" + "=" * 60)
    print(f"ENCODING: {encoding_name}")
    print("=" * 60)

    train_data = spark.read.parquet(train_path).cache()
    test_data = spark.read.parquet(test_path).cache()

    print(f"Train: {train_data.count()}")
    print(f"Test : {test_data.count()}")


    print("\n>>> LOGISTIC REGRESSION")

    start_time = time.time()

    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="label"
    )

    lr_model = lr.fit(train_data)
    lr_model.write().overwrite().save("hdfs://master:9000/DACK/lr_model_best")
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
    train_data.unpersist()
    test_data.unpersist()
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

# INDEX ENCODING

evaluate_dataset(
    "hdfs://master:9000/DACK/weather_ml_rain_index_train",
    "hdfs://master:9000/DACK/weather_ml_rain_index_test",
    "Index Encoding"
)

# ONE-HOT ENCODING

evaluate_dataset(
    "hdfs://master:9000/DACK/weather_ml_rain_ohe_train",
    "hdfs://master:9000/DACK/weather_ml_rain_ohe_test",
    "One-Hot Encoding"
)

# ==========================================
# PHẦN 4: BẢNG KẾT QUẢ
# ==========================================

print("\n")
print("=" * 80)
print("TỔNG HỢP KẾT QUẢ")
print("=" * 80)

results_df = pd.DataFrame(
    results,
    columns=["Encoding", "Model", "ROC_AUC", "F1_Score", "Time_Seconds"]
)
print(results_df)

# ==========================================
# PHẦN 5: XUẤT KẾT QUẢ
# ==========================================

spark_results_df = spark.createDataFrame(
    results,
    schema=["Encoding", "Model", "ROC_AUC", "F1_Score", "Time_Seconds"]
)

output_path = "hdfs://master:9000/DACK/model_comparison_results"

spark_results_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .csv(output_path, header=True)

print("\nĐã lưu thành công lên HDFS tại:")
print(output_path)


# ==========================================
# PHẦN 8: ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT
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