# ================================================================
# FILE: 04_mllib_rain.py — Bài toán 1: Dự đoán mưa ngày mai
# ================================================================
# Bài toán: Binary Classification
#   Input : Các chỉ số thời tiết hôm nay
#   Output: RainTomorrow = Yes / No
#
# Pipeline: StringIndexer → OneHotEncoder → VectorAssembler
#           → StandardScaler → RandomForestClassifier
#
# Thứ tự chạy: Sau 02_preprocessing.py
# ================================================================

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from data_loader import get_spark, load_parquet
import config

# ----------------------------------------------------------------
# KHỞI TẠO SPARK
# ----------------------------------------------------------------
spark = get_spark("04_MLlib_Rain_WeatherAUS")


# ================================================================
# BƯỚC 1: ĐỌC DATA
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 1: ĐỌC DATA ML CHO BÀI TOÁN MƯA")
print("=" * 60)

df = load_parquet(spark, config.ML_RAIN_PATH)
df.persist()

total = df.count()
print(f"\nTổng số mẫu: {total:,}")
print("\nPhân phối biến mục tiêu RainTomorrow:")
df.groupBy("RainTomorrow") \
  .count() \
  .withColumn("Pct_%", F.round(F.col("count") / total * 100, 1)) \
  .show()
print("⚠️  Dữ liệu mất cân bằng ~78%/22% → cần xử lý classWeight")


# ================================================================
# BƯỚC 2: XỬ LÝ MẤT CÂN BẰNG — TÍNH CLASS WEIGHT
# Lý do: Nếu không xử lý, Random Forest sẽ "lười" đoán toàn No
# vì No chiếm 78% → vẫn đúng 78% nhưng không phát hiện được mưa
# Class weight: lớp thiểu số (Yes) được phạt nặng hơn khi sai
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 2: TÍNH CLASS WEIGHT (Xử lý Imbalanced)")
print("=" * 60)

# Đếm số lượng từng class
counts = df.groupBy("RainTomorrow").count().collect()
count_dict = {r["RainTomorrow"]: r["count"] for r in counts}
n_classes  = 2

# Công thức: weight = total / (n_classes × count_of_class)
weight_no  = total / (n_classes * count_dict.get("No",  1))
weight_yes = total / (n_classes * count_dict.get("Yes", 1))

print(f"  Weight cho 'No'  (đa số): {weight_no:.4f}")
print(f"  Weight cho 'Yes' (thiểu số): {weight_yes:.4f}")
print(f"  Tỷ lệ: 'Yes' được coi trọng hơn {weight_yes/weight_no:.1f}x")

# Gán cột classWeight vào DataFrame
df = df.withColumn(
    "classWeight",
    F.when(F.col("RainTomorrow") == "Yes", weight_yes)
     .otherwise(weight_no)
)


# ================================================================
# BƯỚC 3: XÂY DỰNG PIPELINE
# Pipeline = dây chuyền xử lý tự động, theo thứ tự:
#   Chữ → Số thứ tự (StringIndexer)
#   Số thứ tự → Vector 0/1 (OneHotEncoder)
#   Gom tất cả → 1 vector features (VectorAssembler)
#   Chuẩn hóa về cùng thang đo (StandardScaler)
#   Huấn luyện (RandomForestClassifier)
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 3: XÂY DỰNG PIPELINE ML")
print("=" * 60)

cat_features = config.RAIN_CAT_FEATURES
num_features = config.RAIN_NUM_FEATURES

# --- Stage 1: StringIndexer ---
# Chuyển cột chữ thành số thứ tự
# Ví dụ: Location: Sydney→0, Melbourne→1, Perth→2...
# handleInvalid="keep": nếu gặp giá trị mới khi predict → giữ lại, không báo lỗi
print("\nStage 1: StringIndexer (chữ → số thứ tự)")
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"
    )
    for c in cat_features
]
for idx in indexers:
    print(f"  {idx.getInputCol()} → {idx.getOutputCol()}")

# --- Stage 2: OneHotEncoder ---
# Chuyển số thứ tự thành vector 0/1
# Ví dụ: Location có 49 giá trị → 49 cột 0/1
# Tránh model hiểu nhầm: Sydney(0) < Melbourne(1) < Perth(2)
# Thực ra chúng là ngang hàng nhau, không có thứ tự
print("\nStage 2: OneHotEncoder (số thứ tự → vector 0/1)")
encoders = [
    OneHotEncoder(
        inputCol=c + "_idx",
        outputCol=c + "_ohe"
    )
    for c in cat_features
]

# --- Stage 3: VectorAssembler ---
# Gom TẤT CẢ features (số + encoded chữ) thành 1 cột vector duy nhất
# MLlib chỉ chấp nhận input là 1 cột vector, không nhận nhiều cột riêng lẻ
ohe_cols = [c + "_ohe" for c in cat_features]
all_features = num_features + ohe_cols
assembler = VectorAssembler(
    inputCols=all_features,
    outputCol="raw_features",
    handleInvalid="keep"   # Bỏ qua dòng có missing sau khi xử lý
)
print(f"\nStage 3: VectorAssembler → gom {len(all_features)} cột thành 1 vector")

# --- Stage 4: StandardScaler ---
# Chuẩn hóa: đưa tất cả features về cùng thang đo (mean=0, std=1)
# Cần thiết vì: Pressure (1000-1030) vs Cloud (0-8) → chênh lệch 100x
# Nếu không scale: Pressure sẽ "lấn át" Cloud trong tính toán
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,   # Trừ đi mean → dịch về 0
    withStd=True     # Chia cho std → scale về 1
)
print("Stage 4: StandardScaler → chuẩn hóa về mean=0, std=1")

# --- Stage 5: StringIndexer cho Label ---
# Chuyển "Yes"/"No" thành 1.0/0.0 để RandomForest hiểu được
label_indexer = StringIndexer(
    inputCol="RainTomorrow",
    outputCol="label"
)
print("Stage 5: LabelIndexer → RainTomorrow: Yes=1.0, No=0.0")

# --- Stage 6: RandomForestClassifier ---
# Thuật toán: tạo nhiều cây quyết định (Decision Tree) ngẫu nhiên
# mỗi cây học trên tập dữ liệu con → kết quả là đa số phiếu
# Ưu điểm: chịu được dữ liệu lệch, không cần feature selection thủ công
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    weightCol="classWeight",   # Dùng weight để xử lý imbalanced
    numTrees=100,              # 100 cây — đủ ổn định, không quá chậm
    maxDepth=10,               # Độ sâu tối đa mỗi cây
    seed=42                    # Seed cố định → kết quả tái lặp được
)
print("Stage 6: RandomForestClassifier (100 cây, maxDepth=10, classWeight)")

# Ghép tất cả stages vào Pipeline
pipeline = Pipeline(
    stages=indexers + encoders + [assembler, scaler, label_indexer, rf]
)
print(f"\n✅ Pipeline tổng cộng {len(pipeline.getStages())} stages")


# ================================================================
# BƯỚC 4: CHIA TRAIN/TEST VÀ HUẤN LUYỆN
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 4: CHIA TRAIN/TEST (80/20) VÀ HUẤN LUYỆN")
print("=" * 60)

# randomSplit: chia ngẫu nhiên 80% train, 20% test
# seed=42: cố định random seed để kết quả tái lặp được
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"  Tập Train: {train_df.count():,} mẫu (80%)")
print(f"  Tập Test : {test_df.count():,} mẫu (20%)")

# Kiểm tra phân phối target trong tập train
print("\nPhân phối trong tập Train:")
train_df.groupBy("RainTomorrow").count().show()

print("\nĐang huấn luyện Random Forest... (có thể mất vài phút)")
model = pipeline.fit(train_df)
print("✅ Huấn luyện hoàn tất!")


# ================================================================
# BƯỚC 5: DỰ ĐOÁN VÀ ĐÁNH GIÁ
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH")
print("=" * 60)

# model.transform(): áp dụng pipeline lên tập test → ra kết quả dự đoán
predictions = model.transform(test_df)

# --- Accuracy ---
# Tỷ lệ dự đoán đúng / tổng số mẫu
accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

# --- F1-Score ---
# Trung bình điều hòa giữa Precision và Recall
# Tốt hơn Accuracy khi dữ liệu mất cân bằng
f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="f1"
).evaluate(predictions)

# --- ROC-AUC ---
# Diện tích dưới đường cong ROC
# = khả năng model phân biệt đúng Yes vs No
# AUC=0.5: random | AUC=1.0: hoàn hảo | AUC>0.7: chấp nhận được
roc_auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
).evaluate(predictions)

# --- Precision và Recall riêng cho class Yes ---
precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(predictions)

recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(predictions)

print(f"""
╔══════════════════════════════════════════════════╗
║  KẾT QUẢ ĐÁNH GIÁ — DỰ ĐOÁN MƯA NGÀY MAI       ║
╠══════════════════════════════════════════════════╣
║  Accuracy           : {accuracy*100:>6.2f}%                    ║
║  F1-Score (weighted): {f1:>8.4f}                  ║
║  ROC-AUC            : {roc_auc:>8.4f}                  ║
║  Precision (weighted): {precision:>7.4f}                  ║
║  Recall (weighted)  : {recall:>8.4f}                  ║
╚══════════════════════════════════════════════════╝
""")

print("Confusion Matrix (hàng=thực tế, cột=dự đoán):")
print("  label 0 = No (không mưa) | label 1 = Yes (có mưa)")
predictions.groupBy("label", "prediction") \
           .count() \
           .orderBy("label", "prediction") \
           .show()

# --- Feature Importance ---
print("Top 10 Features quan trọng nhất:")
rf_model      = model.stages[-1]   # RandomForest là stage cuối cùng
importances   = rf_model.featureImportances

# Lấy tên features từ VectorAssembler
assembler_stage = model.stages[-4]  # VectorAssembler
feature_names   = assembler_stage.getInputCols()

# Ghép tên và tầm quan trọng
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
model.write().overwrite().save(config.MODEL_RAIN_PATH)
print(f"✅ Đã lưu model_rain → {config.MODEL_RAIN_PATH}")


# ================================================================
# KẾT THÚC
# ================================================================
df.unpersist()
spark.stop()
print("\n✅ Bài toán dự đoán mưa hoàn tất!")
print("   Chạy tiếp: python 04_mllib_fire.py")