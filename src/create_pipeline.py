from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, to_date, coalesce
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

spark = SparkSession.builder.appName("Create_Pipeline").master("local[*]").getOrCreate()

print("1. Dang doc du lieu sach tu buoc tien xu ly...")
df = spark.read.parquet("hdfs://master:9000/DACK/weather_clean")

cat_cols = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
for c in cat_cols:
    if c + "_index" in df.columns:
        df = df.drop(c + "_index")

df = df.withColumn("TempRange", col("MaxTemp") - col("MinTemp")) \
       .withColumn("HumidityDiff", col("Humidity9am") - col("Humidity3pm")) \
       .withColumn("PressureDiff", col("Pressure9am") - col("Pressure3pm")) \
       .withColumn("WindSpeedDiff", col("WindSpeed3pm") - col("WindSpeed9am"))

date_formats = ["yyyy-MM-dd", "dd/MM/yyyy", "M/d/yyyy", "yyyy/MM/dd"]
date_exprs = [to_date(col("Date"), fmt) for fmt in date_formats]

df = df.withColumn("DateParsed", coalesce(*date_exprs)) \
       .withColumn("Month", month(col("DateParsed")))

# Dùng skip cho Indexer để lọc rác
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="skip") for c in cat_cols]

encoder = OneHotEncoder(inputCols=[c + "_index" for c in cat_cols], outputCols=[c + "_ohe" for c in cat_cols], dropLast=True)

feature_cols = [
    "MinTemp", "Temp3pm", "TempRange", "Rainfall",
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    "Pressure9am", "PressureDiff",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "WindSpeedDiff",
    "Month",
    "Location_ohe", "WindGustDir_ohe", "WindDir9am_ohe", "WindDir3pm_ohe", "RainToday_ohe"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

print("2. Dang dong goi Hop den Pipeline...")
stages = indexers + [encoder, assembler, scaler]
pipeline = Pipeline(stages=stages)
df.cache()
pipeline_model = pipeline.fit(df)
df.unpersist()
pipeline_model.write().overwrite().save("hdfs://master:9000/DACK/weather_pipeline_model")
print("THANH CONG! Da luu Pipeline")
spark.stop()