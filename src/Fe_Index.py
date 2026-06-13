from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PHẦN 1: KHỞI TẠO VÀ ĐỌC DỮ LIỆU SẠCH V2
# ==========================================
spark = SparkSession.builder \
    .appName("Weather_Feature_Engineering") \
    .master("local[*]") \
    .getOrCreate()

print("Đang đọc dữ liệu siêu sạch từ bước tiền xử lý...")
df = spark.read.parquet(
    "hdfs://localhost:9000/DACK/weather_clean_v2")
print("Đang tính toán các chỉ số chênh lệch...")

df = df.withColumn("TempRange", col("MaxTemp") - col("MinTemp"))
df = df.withColumn("HumidityDiff", col("Humidity9am") - col("Humidity3pm"))
df = df.withColumn("PressureDiff", col("Pressure9am") - col("Pressure3pm"))
df = df.withColumn("WindSpeedDiff", col("WindSpeed3pm") - col("WindSpeed9am"))

# ==========================================
# PHẦN 3: VECTOR ASSEMBLER (ML PIPELINE)
# ==========================================
print("Đang gom nhóm biến độc lập...")

feature_cols = [
    "MinTemp", "Temp3pm", "TempRange",
    "Rainfall",
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    "Pressure9am", "PressureDiff",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "WindSpeedDiff",
    "Month",
    "Location_index", "WindGustDir_index",
    "WindDir9am_index", "WindDir3pm_index", "RainToday_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df)


# ==========================================
# PHẦN 4: STANDARD SCALER
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
# PHẦN 5: LƯU HDFS
# ==========================================
print("Đang lưu dữ liệu...")

df_save = df_final.select("scaled_features", "label")

hdfs_output_path = "hdfs://localhost:9000/DACK/weather_ml_rain_index"

df_save.coalesce(1) \
    .write \
    .mode("overwrite") \
    .parquet(hdfs_output_path)

print("========= THÀNH CÔNG =========")
print(f"Saved to: {hdfs_output_path}")