from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder

# ==========================================
# PHẦN 1: KHỞI TẠO VÀ ĐỌC DỮ LIỆU SẠCH V2
# ==========================================
spark = SparkSession.builder \
    .appName("Weather_Feature_Engineering_OHE") \
    .master("local[*]") \
    .getOrCreate()

print("Đang đọc dữ liệu sạch từ bước tiền xử lý...")

df = spark.read.csv(
    "weather_clean_v2",
    header=True,
    inferSchema=True
)

print(f"Số dòng dữ liệu: {df.count()}")

# ==========================================
# PHẦN 2: FEATURE ENGINEERING
# ==========================================
print("Đang tạo các đặc trưng mới...")

# Biên độ nhiệt độ
df = df.withColumn(
    "TempRange",
    col("MaxTemp") - col("MinTemp")
)

# Chênh lệch độ ẩm
df = df.withColumn(
    "HumidityDiff",
    col("Humidity9am") - col("Humidity3pm")
)

# Chênh lệch áp suất
df = df.withColumn(
    "PressureDiff",
    col("Pressure9am") - col("Pressure3pm")
)

# Chênh lệch tốc độ gió
df = df.withColumn(
    "WindSpeedDiff",
    col("WindSpeed3pm") - col("WindSpeed9am")
)

# ==========================================
# PHẦN 3: ONE-HOT ENCODING
# ==========================================
print("Đang thực hiện One-Hot Encoding...")

index_cols = [
    "Location_index",
    "WindGustDir_index",
    "WindDir9am_index",
    "WindDir3pm_index",
    "RainToday_index"
]

ohe_cols = [
    "Location_ohe",
    "WindGustDir_ohe",
    "WindDir9am_ohe",
    "WindDir3pm_ohe",
    "RainToday_ohe"
]

encoder = OneHotEncoder(
    inputCols=index_cols,
    outputCols=ohe_cols,
    dropLast=True
)

encoder_model = encoder.fit(df)
df = encoder_model.transform(df)

print("Kiểm tra kết quả One-Hot Encoding:")
df.select(
    "Location",
    "Location_index",
    "Location_ohe"
).show(5, truncate=False)

# ==========================================
# PHẦN 4: VECTOR ASSEMBLER
# ==========================================
print("Đang gom các đặc trưng thành vector features...")

feature_cols = [

    # Nhóm nhiệt độ
    "MinTemp",
    "MaxTemp",
    "Temp9am",
    "Temp3pm",
    "TempRange",

    # Lượng mưa
    "Rainfall",

    # Độ ẩm
    "Humidity9am",
    "Humidity3pm",
    "HumidityDiff",

    # Áp suất
    "Pressure9am",
    "Pressure3pm",
    "PressureDiff",

    # Gió
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "WindSpeedDiff",

    # Thời gian
    "Month",

    # One-Hot Encoded Features
    "Location_ohe",
    "WindGustDir_ohe",
    "WindDir9am_ohe",
    "WindDir3pm_ohe",
    "RainToday_ohe"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_vector = assembler.transform(df)

# ==========================================
# PHẦN 5: STANDARD SCALER
# ==========================================
print("Đang chuẩn hóa dữ liệu...")

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=False
)

scaler_model = scaler.fit(df_vector)

df_final = scaler_model.transform(df_vector)

# ==========================================
# PHẦN 6: CHỌN CỘT PHỤC VỤ MACHINE LEARNING
# ==========================================
df_save = df_final.select(
    "scaled_features",
    "label"
)

print("Kiểm tra dữ liệu cuối cùng:")
df_save.show(5, truncate=False)

# ==========================================
# PHẦN 7: LƯU LÊN HDFS
# ==========================================
hdfs_output_path = "hdfs://localhost:9000/DACK/weather_ml_rain_ohe"

print("Đang lưu dữ liệu lên HDFS...")

df_save.coalesce(1) \
    .write \
    .mode("overwrite") \
    .parquet(hdfs_output_path)

print("\n===================================")
print("THÀNH CÔNG!")
print("Dataset One-Hot Encoding đã được lưu tại:")
print(hdfs_output_path)
print("===================================")

spark.stop()