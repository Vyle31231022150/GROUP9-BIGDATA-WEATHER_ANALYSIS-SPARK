# ================================================================
# FILE: 04_mllib_fire.py — Bài toán 2: Phân loại nguy cơ cháy rừng
# ================================================================
# Bài toán: Multiclass Classification (4 lớp)
#   Input : Các chỉ số thời tiết (đã DROP 3 cột tạo nhãn)
#   Output: Bushfire_Risk_Level = 0/1/2/3
#           0=Low | 1=Moderate | 2=High | 3=Extreme
#
# Kỹ thuật đặc biệt:
#   ✅ Synthetic Labeling: nhãn được tạo từ Business Rules
#   ✅ Data Leakage Prevention: drop MaxTemp, Humidity3pm, WindGustSpeed
#   ✅ Class Weight: xử lý mất cân bằng lớp nghiêm trọng
#
# Thứ tự chạy: Sau 02_preprocessing.py (và sau 04_mllib_rain.py)
# ================================================================

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from data_loader import get_spark, load_parquet
import config

# ----------------------------------------------------------------
# KHỞI TẠO SPARK
# ----------------------------------------------------------------
spark = get_spark("04_MLlib_Fire_WeatherAUS")


# ================================================================
# BƯỚC 1: ĐỌC DATA VÀ KIỂM TRA
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 1: ĐỌC DATA ML CHO BÀI TOÁN CHÁY RỪNG")
print("=" * 60)

df = load_parquet(spark, config.ML_FIRE_PATH)
df.persist()

total = df.count()
print(f"\nTổng số mẫu: {total:,}")

# Xác nhận 3 cột tạo nhãn đã bị xóa từ bước preprocessing
print("\n✅ Kiểm tra Data Leakage Prevention:")
for col_name in config.FIRE_LEAKAGE_COLS:
    exists = col_name in df.columns
    status = "❌ VẪN CÒN — LỖI LEAKAGE!" if exists else "✅ Đã xóa"
    print(f"  {col_name}: {status}")

print("\nCác cột hiện có trong dataset:")
print(df.columns)

print("\nPhân phối nhãn Bushfire_Risk_Level:")
dist = df.groupBy("Bushfire_Risk_Level") \
         .count() \
         .orderBy("Bushfire_Risk_Level") \
         .withColumn("Pct_%", F.round(F.col("count") / total * 100, 2)) \
         .withColumn("Nhan",
             F.when(F.col("Bushfire_Risk_Level") == 0, "Low")
              .when(F.col("Bushfire_Risk_Level") == 1, "Moderate")
              .when(F.col("Bushfire_Risk_Level") == 2, "High")
              .otherwise("Extreme")
         )
dist.show()


# ================================================================
# BƯỚC 2: TÍNH CLASS WEIGHT
# Cháy rừng Extreme (class 3) rất hiếm → cần weight cao để model
# không bỏ qua lớp này khi huấn luyện
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 2: TÍNH CLASS WEIGHT (Xử lý Imbalanced)")
print("=" * 60)

counts = df.groupBy("Bushfire_Risk_Level").count().collect()
count_dict = {r["Bushfire_Risk_Level"]: r["count"] for r in counts}
n_classes  = 4  # 4 lớp: 0, 1, 2, 3

weights = {}
for cls in range(n_classes):
    cnt = count_dict.get(cls, 1)
    # Lớp ít mẫu → weight cao → được chú ý nhiều hơn khi train
    weights[cls] = total / (n_classes * cnt)

print(f"{'Class':<10} {'Tên':>12} {'Số mẫu':>10} {'Weight':>10}")
print("-" * 45)
for cls in range(n_classes):
    names = {0:"Low", 1:"Moderate", 2:"High", 3:"Extreme"}
    print(f"  {cls:<8} {names[cls]:>12} "
          f"{count_dict.get(cls,0):>10,} {weights[cls]:>10.4f}")

# Tạo cột classWeight trong DataFrame
df = df.withColumn(
    "classWeight",
    F.when(F.col("Bushfire_Risk_Level") == 0, weights[0])
     .when(F.col("Bushfire_Risk_Level") == 1, weights[1])
     .when(F.col("Bushfire_Risk_Level") == 2, weights[2])
     .otherwise(weights[3])
)

# Chuyển label thành Double (MLlib yêu cầu)
df = df.withColumn(
    "label",
    F.col("Bushfire_Risk_Level").cast("double")
)


# ================================================================
# BƯỚC 3: XÂY DỰNG PIPELINE
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 3: XÂY DỰNG PIPELINE ML (CHÁY RỪNG)")
print("=" * 60)

cat_features = config.FIRE_CAT_FEATURES
num_features = config.FIRE_NUM_FEATURES

print(f"Features số  : {len(num_features)} cột")
print(f"Features chữ : {len(cat_features)} cột")
print(f"Tổng features: {len(num_features) + len(cat_features)} (trước encoding)")
print("\n⚠️  Lưu ý: MaxTemp, Humidity3pm, WindGustSpeed đã bị loại bỏ")
print("   để ngăn Data Leakage — model phải học từ features còn lại")

# StringIndexer cho cột chữ
indexers = [
    StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
    for c in cat_features
]

# OneHotEncoder
encoders = [
    OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_ohe")
    for c in cat_features
]

# VectorAssembler
ohe_cols  = [c + "_ohe" for c in cat_features]
assembler = VectorAssembler(
    inputCols=num_features + ohe_cols,
    outputCol="raw_features",
    handleInvalid="keep"
)

# StandardScaler
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True, withStd=True
)

# RandomForestClassifier cho 4 lớp
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    weightCol="classWeight",
    numTrees=150,       # Nhiều cây hơn vì multiclass phức tạp hơn
    maxDepth=12,        # Sâu hơn để học ranh giới 4 lớp
    numClasses=4,
    seed=42
)

pipeline = Pipeline(
    stages=indexers + encoders + [assembler, scaler, rf]
)
print(f"\n✅ Pipeline: {len(pipeline.getStages())} stages")


# ================================================================
# BƯỚC 4: TRAIN / TEST VÀ HUẤN LUYỆN
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 4: CHIA TRAIN/TEST (80/20) VÀ HUẤN LUYỆN")
print("=" * 60)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"  Tập Train: {train_df.count():,} mẫu")
print(f"  Tập Test : {test_df.count():,} mẫu")

print("\nPhân phối nhãn trong tập Train:")
train_df.groupBy("Bushfire_Risk_Level").count().orderBy("Bushfire_Risk_Level").show()

print("Đang huấn luyện... (có thể mất vài phút với 150 cây)")
model = pipeline.fit(train_df)
print("✅ Huấn luyện hoàn tất!")


# ================================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH CHÁY RỪNG")
print("=" * 60)

predictions = model.transform(test_df)

# Accuracy
accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

# F1 weighted — phù hợp hơn với multiclass mất cân bằng
f1_weighted = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="weightedFMeasure"
).evaluate(predictions)

# Precision weighted
precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(predictions)

# Recall weighted
recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(predictions)

print(f"""
╔══════════════════════════════════════════════════╗
║  KẾT QUẢ — PHÂN LOẠI NGUY CƠ CHÁY RỪNG          ║
╠══════════════════════════════════════════════════╣
║  Accuracy            : {accuracy*100:>6.2f}%                   ║
║  F1-Score (weighted) : {f1_weighted:>8.4f}                 ║
║  Precision (weighted): {precision:>8.4f}                 ║
║  Recall (weighted)   : {recall:>8.4f}                 ║
╚══════════════════════════════════════════════════╝
""")

# Confusion Matrix chi tiết
print("Confusion Matrix:")
print("  (Hàng = Nhãn thực | Cột = Nhãn dự đoán)")
predictions.groupBy("label", "prediction") \
           .count() \
           .orderBy("label", "prediction") \
           .show()

# Phân tích kết quả từng class
print("Chi tiết accuracy từng cấp độ nguy cơ:")
print(f"{'Class':<5} {'Tên':>12} {'Đúng':>8} {'Tổng':>8} {'Accuracy':>10}")
print("-" * 50)
for cls, name in config.FIRE_RISK_LABELS.items():
    subset  = predictions.filter(F.col("label") == float(cls))
    total_c = subset.count()
    if total_c == 0:
        continue
    correct = subset.filter(
        F.col("prediction") == float(cls)
    ).count()
    acc_cls = correct / total_c * 100
    print(f"  {cls:<3} {name:>20} {correct:>8,} {total_c:>8,} {acc_cls:>9.1f}%")

# Feature Importance
print("\nTop 10 Features quan trọng nhất (theo mô hình cháy rừng):")
rf_model      = model.stages[-1]
importances   = rf_model.featureImportances
assembler_stg = model.stages[-3]
feature_names = assembler_stg.getInputCols()

feat_imp = sorted(
    zip(feature_names, importances.toArray()),
    key=lambda x: x[1], reverse=True
)[:10]

print(f"\n{'Feature':<30} {'Importance':>12}  Biểu đồ")
print("-" * 60)
for name, imp in feat_imp:
    bar = "█" * int(imp * 100)
    print(f"{name:<30} {imp:>10.4f}  {bar}")


# ================================================================
# BƯỚC 6: LƯU MODEL
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 6: LƯU MODEL VÀO HDFS")
print("=" * 60)
model.write().overwrite().save(config.MODEL_FIRE_PATH)
print(f"✅ Đã lưu model_fire → {config.MODEL_FIRE_PATH}")


# ================================================================
# KẾT THÚC
# ================================================================
df.unpersist()
spark.stop()
print("\n✅ Bài toán cháy rừng hoàn tất!")
print("   Chạy tiếp: python 05_streaming.py")