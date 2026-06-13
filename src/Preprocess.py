from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, month, year, coalesce
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# ==========================================
# PHẦN 1: KHỞI TẠO VÀ ĐỌC DỮ LIỆU
# ==========================================
spark = SparkSession.builder \
    .appName("Data_Preprocessing_V2") \
    .master("local[*]") \
    .getOrCreate()

print("Đang nạp dữ liệu gốc...")
# Nhớ kiểm tra lại đường dẫn HDFS nếu cần
df = spark.read.csv("hdfs://master:9000/DACK/weatherAUS.csv", header=True, inferSchema=True, nullValue="NA")
print(f"Tổng số dòng ban đầu: {df.count()}")


# ==========================================
# PHẦN 2: XỬ LÝ MISSING VALUE CHUẨN NGHIỆP VỤ
# ==========================================
# Nhát cắt 1: Bỏ các dòng thiếu đáp án (RainTomorrow)
df = df.dropna(subset=["RainTomorrow"])
print(f"Số dòng sau khi xóa NA của RainTomorrow: {df.count()}")

# Nhát cắt 2: Bỏ các cột thiếu quá nhiều dữ liệu (gây nhiễu)
columns_to_drop = ["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]
df = df.drop(*columns_to_drop)

# Khai báo nhóm cột
numeric_cols = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"
]
categorical_cols = [
    "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"
]

# Nhát cắt 3: Điền Biến Số bằng Median (Trung vị)
print("Đang xử lý Median cho các cột số...")
for c in numeric_cols:
    median_val = df.approxQuantile(c, [0.5], 0.0)[0]
    df = df.fillna({c: median_val})

# Nhát cắt 4: Điền Biến Phân loại bằng Mode (Xuất hiện nhiều nhất)
print("Đang xử lý Mode cho các cột phân loại...")
for c in categorical_cols:
    mode_row = df.filter(col(c).isNotNull()).groupBy(c).count().orderBy(col("count").desc()).first()
    if mode_row is not None:
        df = df.fillna({c: mode_row[0]})


# ==========================================
# PHẦN 3: MÃ HÓA NHÃN & XỬ LÝ THỜI GIAN
# ==========================================
print("Đang mã hóa các biến phân loại thành số (StringIndexer)...")
# 1. Mã hóa RainTomorrow thành 'label' (Vì chỉ còn Yes/No nên sẽ sinh ra đúng 0.0 và 1.0)
label_indexer = StringIndexer(inputCol="RainTomorrow", outputCol="label")
df = label_indexer.fit(df).transform(df)

# 2. Mã hóa các biến categorical khác
for c in categorical_cols:
    indexer = StringIndexer(inputCol=c, outputCol=c + "_index")
    df = indexer.fit(df).transform(df)

# 3. Tách tháng và năm từ cột Date
print("Đang tách thông tin ngày tháng...")
df = df.withColumn(
    "DateParsed",
    coalesce(
        to_timestamp(col("Date"), "d/M/yyyy"),     # 1. Thử chuẩn Việt Nam/Úc
        to_timestamp(col("Date"), "yyyy-MM-dd"),   # 2. Thử chuẩn Quốc tế ISO
        to_timestamp(col("Date"), "M/d/yyyy")      # 3. Thử chuẩn Mỹ 
    )
) \
.withColumn("Month", month(col("DateParsed"))) \
.withColumn("Year", year(col("DateParsed")))

# ==========================================
# PHẦN 4: LƯU DỮ LIỆU SẠCH LÊN HDFS
# ==========================================

print("Đang lưu dữ liệu sạch lên HDFS...")

hdfs_output_path = "hdfs://master:9000/DACK/weather_clean"

df.write \
    .mode("overwrite") \
    .parquet(hdfs_output_path)

print("========= THÀNH CÔNG =========")
print(f"Saved to: {hdfs_output_path}")
