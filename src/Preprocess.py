from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, month, year, coalesce

# ==========================================
# PHẦN 1: KHỞI TẠO SPARK
# ==========================================

spark = SparkSession.builder \
    .appName("Weather_Preprocessing_Clean") \
    .master("local[*]") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# ==========================================
# PHẦN 2: ĐỌC DỮ LIỆU GỐC
# ==========================================

print("Đang đọc dữ liệu gốc...")

df = spark.read.csv(
    "hdfs://master:9000/DACK/weatherAUS.csv",
    header=True,
    inferSchema=True,
    nullValue="NA"
)

print(f"Total rows: {df.count()}")

# ==========================================
# PHẦN 3: XỬ LÝ MISSING VALUES
# ==========================================

print("Xử lý missing values...")

# bỏ target null
df = df.dropna(subset=["RainTomorrow"])

# bỏ cột quá nhiều missing
columns_to_drop = [
    "Evaporation",
    "Sunshine",
    "Cloud9am",
    "Cloud3pm"
]

df = df.drop(*columns_to_drop)
df.cache()

# numeric cols
numeric_cols = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm"
]

# categorical cols
categorical_cols = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday"
]

# fill median
print("Điền giá trị bằng trung vị...")
for c in numeric_cols:
    median_val = df.approxQuantile(c, [0.5], 0.0)[0]
    df = df.fillna({c: median_val})

# fill mode
print("Điền giá trị bằng yếu vị...")
for c in categorical_cols:
    mode_row = df.filter(col(c).isNotNull()) \
        .groupBy(c) \
        .count() \
        .orderBy(col("count").desc()) \
        .first()

    if mode_row is not None:
        df = df.fillna({c: mode_row[0]})

# ==========================================
# PHẦN 4: XỬ LÝ DATE
# ==========================================

print("Xử lý dữ liệu Ngày tháng...")

date_formats = [
    "yyyy-MM-dd", "dd/MM/yyyy", "MM/dd/yyyy",
    "d/M/yyyy", "M/d/yyyy", "yyyy/MM/dd", "dd-MM-yyyy"
]
date_exprs = [to_timestamp(col("Date"), fmt) for fmt in date_formats]

df = df.withColumn("DateParsed", coalesce(*date_exprs))

df = df.withColumn("Month", month(col("DateParsed"))) \
       .withColumn("Year", year(col("DateParsed")))

# ==========================================
# PHẦN 5: SAVE CLEAN DATASET
# ==========================================

print("Lưu bộ dữ liệu được làm sạch lên hdfs...")

output_path = "hdfs://master:9000/DACK/weather_clean"

df.write \
    .mode("overwrite") \
    .parquet(output_path)

print("DONE")
print(f"Saved to: {output_path}")
df.unpersist()

spark.stop()