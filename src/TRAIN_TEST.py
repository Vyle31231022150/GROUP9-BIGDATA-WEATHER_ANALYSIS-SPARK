from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Train_Test_Split") \
    .master("local[*]") \
    .getOrCreate()


# ==========================================
# INDEX DATASET
# ==========================================

print("Đang đọc Index Dataset...")

df_index = spark.read.parquet(
    "hdfs://master:9000/DACK/weather_ml_rain_index"
)

train_index, test_index = df_index.randomSplit(
    [0.7, 0.3],
    seed=42
)

train_index.write \
    .mode("overwrite") \
    .parquet("hdfs://master:9000/DACK/weather_ml_rain_index_train")

test_index.write \
    .mode("overwrite") \
    .parquet("hdfs://master:9000/DACK/weather_ml_rain_index_test")

print("Đã lưu Index Train/Test")

# ==========================================
# OHE DATASET
# ==========================================

print("Đang đọc OHE Dataset...")

df_ohe = spark.read.parquet(
    "hdfs://master:9000/DACK/weather_ml_rain_ohe"
)

train_ohe, test_ohe = df_ohe.randomSplit(
    [0.7, 0.3],
    seed=42
)

train_ohe.write \
    .mode("overwrite") \
    .parquet("hdfs://master:9000/DACK/weather_ml_rain_ohe_train")

test_ohe.write \
    .mode("overwrite") \
    .parquet("hdfs://master:9000/DACK/weather_ml_rain_ohe_test")

print("Đã lưu OHE Train/Test")

spark.stop()