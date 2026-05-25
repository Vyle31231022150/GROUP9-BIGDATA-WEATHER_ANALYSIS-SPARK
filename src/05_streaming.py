# ================================================================
# FILE: 05_streaming.py — Spark Structured Streaming
# ================================================================
# ⭐ ĐIỂM CỘNG +4: Kỹ thuật Streaming (mục 2.5 đề bài)
#
# Mục đích:
#   Giả lập hệ thống dự đoán thời tiết REALTIME:
#   → Dữ liệu mới (giả lập từng batch nhỏ) liên tục đổ vào
#   → Spark tự động phát hiện và xử lý từng batch ngay lập tức
#   → Load model đã train để dự đoán: mưa ngày mai + nguy cơ cháy rừng
#
# Tại sao gọi là "Streaming"?
#   Bình thường (batch): đọc toàn bộ file → xử lý một lần
#   Streaming: đọc liên tục từng mảnh → xử lý ngay khi đến
#   Giống như xem TV trực tiếp vs xem phim đã quay sẵn
#
# Cách giả lập:
#   1. Chia weatherAUS thành nhiều file nhỏ (mỗi file = 1 "batch")
#   2. Spark theo dõi thư mục, thấy file mới → xử lý ngay
#   3. Trong thực tế: thay bằng Kafka, socket, hay IoT sensor
#
# Thứ tự chạy: Sau 04_mllib_rain.py và 04_mllib_fire.py
# ================================================================

import os
import shutil
import time
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from data_loader import get_spark, load_parquet
import config

# ----------------------------------------------------------------
# KHỞI TẠO SPARK
# ----------------------------------------------------------------
spark = get_spark("05_Streaming_WeatherAUS")


# ================================================================
# BƯỚC 1: LOAD CÁC MODEL ĐÃ TRAIN
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 1: LOAD MODEL TỪ HDFS")
print("=" * 60)

# PipelineModel.load(): đọc lại toàn bộ pipeline đã train
# Bao gồm: StringIndexer đã fit, OneHotEncoder, Scaler, RandomForest
print("Đang load model dự đoán mưa...")
model_rain = PipelineModel.load(config.MODEL_RAIN_PATH)
print(f"✅ model_rain: {len(model_rain.stages)} stages")

print("Đang load model cháy rừng...")
model_fire = PipelineModel.load(config.MODEL_FIRE_PATH)
print(f"✅ model_fire: {len(model_fire.stages)} stages")


# ================================================================
# BƯỚC 2: CHUẨN BỊ DỮ LIỆU STREAMING
# Chia file gốc thành nhiều file CSV nhỏ → giả lập dữ liệu liên tục đến
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 2: CHUẨN BỊ DỮ LIỆU STREAMING")
print("=" * 60)

# Tạo/dọn thư mục
for d in [config.STREAM_INPUT_DIR, config.STREAM_OUTPUT_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
print(f"✅ Đã tạo thư mục:\n   Input : {config.STREAM_INPUT_DIR}")
print(f"   Output: {config.STREAM_OUTPUT_DIR}")

# Đọc weather_clean từ HDFS, lấy mẫu 500 dòng để giả lập
df_source = load_parquet(spark, config.CLEAN_DATA_PATH)
df_sample = df_source.limit(500).toPandas()

# Chia thành 10 file, mỗi file 50 dòng = 10 "batch" streaming
BATCH_SIZE   = 50
NUM_BATCHES  = 10
print(f"\nChia {len(df_sample)} dòng thành {NUM_BATCHES} batch "
      f"({BATCH_SIZE} dòng/batch)...")

for i in range(NUM_BATCHES):
    chunk    = df_sample.iloc[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
    filename = f"{config.STREAM_INPUT_DIR}/batch_{i:02d}.csv"
    chunk.to_csv(filename, index=False)

print(f"✅ Đã tạo {NUM_BATCHES} file CSV trong {config.STREAM_INPUT_DIR}")


# ================================================================
# BƯỚC 3: ĐỊNH NGHĨA SCHEMA CHO STREAM
# Spark Streaming cần biết trước cấu trúc dữ liệu (schema)
# vì nó xử lý từng file ngay khi xuất hiện, không có thời gian inferSchema
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 3: ĐỊNH NGHĨA SCHEMA")
print("=" * 60)

# Schema phải khớp với cột trong weather_clean
stream_schema = StructType([
    StructField("Date",          DateType(),   True),
    StructField("Location",      StringType(), True),
    StructField("MinTemp",       DoubleType(), True),
    StructField("MaxTemp",       DoubleType(), True),
    StructField("Rainfall",      DoubleType(), True),
    StructField("Evaporation",   DoubleType(), True),
    StructField("Sunshine",      DoubleType(), True),
    StructField("WindGustDir",   StringType(), True),
    StructField("WindGustSpeed", DoubleType(), True),
    StructField("WindDir9am",    StringType(), True),
    StructField("WindDir3pm",    StringType(), True),
    StructField("WindSpeed9am",  DoubleType(), True),
    StructField("WindSpeed3pm",  DoubleType(), True),
    StructField("Humidity9am",   DoubleType(), True),
    StructField("Humidity3pm",   DoubleType(), True),
    StructField("Pressure9am",   DoubleType(), True),
    StructField("Pressure3pm",   DoubleType(), True),
    StructField("Cloud9am",      DoubleType(), True),
    StructField("Cloud3pm",      DoubleType(), True),
    StructField("Temp9am",       DoubleType(), True),
    StructField("Temp3pm",       DoubleType(), True),
    StructField("RainToday",     StringType(), True),
    StructField("RainTomorrow",  StringType(), True),
    StructField("Year",          IntegerType(),True),
    StructField("Month",         IntegerType(),True),
    StructField("Quarter",       IntegerType(),True),
    StructField("Season",        StringType(), True),
])
print(f"✅ Schema định nghĩa: {len(stream_schema.fields)} cột")


# ================================================================
# BƯỚC 4: TẠO STREAMING DATAFRAME
# readStream: khác với read thông thường
# → Spark theo dõi thư mục liên tục
# → Mỗi khi có file mới → tự động xử lý
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 4: KHỞI TẠO STREAMING DATAFRAME")
print("=" * 60)

stream_df = (
    spark.readStream
    .schema(stream_schema)
    # Mỗi lần trigger chỉ đọc tối đa 1 file
    # → Giả lập từng batch đến theo thứ tự
    .option("maxFilesPerTrigger", 1)
    .csv(config.STREAM_INPUT_DIR, header=True)
)

print("✅ Streaming DataFrame đã sẵn sàng")
print("   → Spark đang theo dõi:", config.STREAM_INPUT_DIR)


# ================================================================
# BƯỚC 5: HÀM XỬ LÝ TỪNG BATCH
# foreachBatch: hàm này được gọi mỗi khi có batch mới đến
# Trong hàm này ta load model và dự đoán
# ================================================================

batch_counter = [0]  # Dùng list để có thể thay đổi trong nested function

def process_batch(batch_df, batch_id):
    """
    Hàm xử lý mỗi micro-batch khi có dữ liệu mới đến.

    Tham số:
        batch_df: DataFrame của batch hiện tại
        batch_id: ID batch (tự động tăng 0, 1, 2...)
    """
    batch_counter[0] += 1
    n_rows = batch_df.count()

    if n_rows == 0:
        print(f"  [Batch {batch_id}] Rỗng — bỏ qua")
        return

    print(f"\n{'─'*60}")
    print(f"  📡 BATCH {batch_id} NHẬN ĐƯỢC: {n_rows} bản ghi mới")
    print(f"{'─'*60}")

    # ---- Dự đoán 1: Mưa ngày mai ----
    print("  🌧️  Dự đoán mưa ngày mai:")
    pred_rain = model_rain.transform(batch_df)
    pred_rain.select(
        "Location",
        F.date_format("Date", "dd/MM/yyyy").alias("Ngay"),
        F.col("Humidity3pm").alias("DoAm_3pm"),
        F.col("RainToday").alias("MuaHomNay"),
        F.when(F.col("prediction") == 1.0, "✅ CÓ MƯA")
         .otherwise("⬜ KHÔNG MƯA")
         .alias("DuDoan_MuaNgayMai")
    ).show(5, truncate=False)

    # ---- Dự đoán 2: Nguy cơ cháy rừng ----
    print("  🔥  Dự đoán nguy cơ cháy rừng:")
    # Phải DROP 3 cột tạo nhãn trước khi đưa vào model_fire
    batch_fire = batch_df.drop(*config.FIRE_LEAKAGE_COLS)
    pred_fire  = model_fire.transform(batch_fire)
    pred_fire.select(
        "Location",
        F.date_format("Date", "dd/MM/yyyy").alias("Ngay"),
        F.col("prediction").cast("int").alias("Risk_Level"),
        F.when(F.col("prediction") == 0, "🟢 Low")
         .when(F.col("prediction") == 1, "🟡 Moderate")
         .when(F.col("prediction") == 2, "🟠 High")
         .otherwise("🔴 Extreme")
         .alias("Muc_Do_Nguy_Co")
    ).show(5, truncate=False)

    # Lưu kết quả ra file
    (
        pred_rain.select(
            "Location", "Date",
            "RainTomorrow",
            F.col("prediction").alias("Rain_Predicted"),
        )
        .write
        .mode("append")
        .parquet(config.STREAM_OUTPUT_DIR + "/rain_predictions")
    )

    print(f"  ✅ Batch {batch_id} xử lý xong → "
          f"đã lưu vào {config.STREAM_OUTPUT_DIR}")


# ================================================================
# BƯỚC 6: KHỞI CHẠY STREAMING QUERY
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 6: KHỞI CHẠY STREAMING")
print("  ⭐ ĐIỂM CỘNG +4: Spark Structured Streaming")
print("=" * 60)
print("Streaming đang chạy... Spark sẽ xử lý từng batch tự động")
print(f"Mỗi 5 giây kiểm tra thư mục {config.STREAM_INPUT_DIR} một lần\n")

# writeStream: khởi chạy streaming
# foreachBatch: gọi hàm process_batch với mỗi micro-batch
# trigger: 5 giây kiểm tra một lần
# outputMode("append"): chỉ ghi dữ liệu mới, không ghi lại cũ
query = (
    stream_df.writeStream
    .foreachBatch(process_batch)
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
)

# awaitTermination(timeout): chạy tối đa 90 giây rồi tự dừng
# Trong thực tế: dùng query.awaitTermination() không có timeout
# → chạy vô hạn cho đến khi dừng thủ công
print("⏳ Streaming chạy trong 90 giây...")
query.awaitTermination(timeout=90)

print("\n✅ Streaming hoàn tất!")
print(f"   Đã xử lý {batch_counter[0]} batch")
print(f"   Kết quả lưu tại: {config.STREAM_OUTPUT_DIR}")


# ================================================================
# BƯỚC 7: ĐỌC LẠI KẾT QUẢ ĐỂ KIỂM TRA
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 7: TỔNG KẾT KẾT QUẢ STREAMING")
print("=" * 60)

try:
    results = spark.read.parquet(
        config.STREAM_OUTPUT_DIR + "/rain_predictions"
    )
    print(f"Tổng số dự đoán đã lưu: {results.count():,}")
    print("\nMẫu kết quả:")
    results.show(10, truncate=False)

    print("\nPhân phối kết quả dự đoán:")
    results.groupBy("Rain_Predicted") \
           .count() \
           .withColumn("Nhan",
               F.when(F.col("Rain_Predicted") == 1.0, "CÓ MƯA")
                .otherwise("KHÔNG MƯA")
           ).show()
except Exception as e:
    print(f"Chưa có kết quả: {e}")


# ================================================================
# KẾT THÚC
# ================================================================
spark.stop()

print("""
✅ TOÀN BỘ PROJECT HOÀN TẤT!

Thứ tự đã thực hiện:
  01_eda.py          → Phân tích khám phá dữ liệu (8 giai đoạn)
  02_preprocessing.py → Tiền xử lý + tạo 3 file data
  03_spark_sql.py    → 10 câu truy vấn Spark SQL phức tạp
  04_mllib_rain.py   → Bài toán 1: Dự đoán mưa ngày mai
  04_mllib_fire.py   → Bài toán 2: Phân loại nguy cơ cháy rừng
  05_streaming.py    → Streaming realtime (⭐ +4 điểm)

Điểm cộng đã áp dụng:
  ✅ +3: persist(), cache(), explain() trong 01_eda.py, 03_spark_sql.py
  ✅ +4: Spark Structured Streaming trong 05_streaming.py
  📌 +3: Multi-node Cluster → đổi SPARK_MASTER trong config.py
""")