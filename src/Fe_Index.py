from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, month, year, coalesce
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer

# ==========================================
# PHẦN 1: START SESSION
# ==========================================

spark = SparkSession.builder \
    .appName("Weather_Feature_Engineering_Index") \
    .master("local[*]") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

print("Đang đọc dữ liệu đã qua tiền xử lý...")

df = spark.read.parquet("hdfs://master:9000/DACK/weather_clean")
df.cache()
print("Tải xong data")

# ==========================================
# PHẦN 2: ENCODING
# ==========================================

print("Đang mã hóa categorical features...")

categorical_cols = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday"
]

# label encoding for target feature
label_indexer = StringIndexer(
    inputCol="RainTomorrow",
    outputCol="label"
)

df = label_indexer.fit(df).transform(df)

# encode categorical features
for c in categorical_cols:
    indexer = StringIndexer(
        inputCol=c,
        outputCol=c + "_index"
    )
    df = indexer.fit(df).transform(df)

# ==========================================
# PHẦN 3: TIME FEATURE
# ==========================================

print("Xử lý Date...")

date_formats = [
    "yyyy-MM-dd", "dd/MM/yyyy", "MM/dd/yyyy",
    "d/M/yyyy", "M/d/yyyy", "yyyy/MM/dd", "dd-MM-yyyy"
]
date_exprs = [to_timestamp(col("Date"), fmt) for fmt in date_formats]

df = df.withColumn("DateParsed", coalesce(*date_exprs)) \
       .withColumn("Month", month(col("DateParsed"))) \
       .withColumn("Year", year(col("DateParsed")))

# ==========================================
# PHẦN 4: FEATURE ENGINEERING
# ==========================================

print("Tạo feature mới...")

df = df.withColumn("TempRange", col("MaxTemp") - col("MinTemp"))
df = df.withColumn("HumidityDiff", col("Humidity9am") - col("Humidity3pm"))
df = df.withColumn("PressureDiff", col("Pressure9am") - col("Pressure3pm"))
df = df.withColumn("WindSpeedDiff", col("WindSpeed3pm") - col("WindSpeed9am"))

# ==========================================
# PHẦN 5: VECTOR ASSEMBLER
# ==========================================

print("Vectorizing features...")

feature_cols = [
    "MinTemp", "Temp3pm", "TempRange",
    "Rainfall",
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    "Pressure9am", "PressureDiff",
    "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "WindSpeedDiff",
    "Month",
    "Location_index",
    "WindGustDir_index",
    "WindDir9am_index",
    "WindDir3pm_index",
    "RainToday_index"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_vector = assembler.transform(df)

# ==========================================
# PHẦN 6: SCALING
# ==========================================

print("Scaling features...")

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=False
)

scaler_model = scaler.fit(df_vector)
df_final = scaler_model.transform(df_vector)

# ==========================================
# PHẦN 7: SAVE INDEX DATASET
# ==========================================

print("Saving dataset...")

df_save = df_final.select("scaled_features", "label")

output_path = "hdfs://master:9000/DACK/weather_ml_rain_index"

df_save.coalesce(1) \
    .write \
    .mode("overwrite") \
    .parquet(output_path)

print("DONE")
print(f"Saved to: {output_path}")
df.unpersist()

spark.stop()