from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_date, coalesce, month, when, get_json_object
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("Weather_Streaming") \
    .master("local[*]") \
    .config("spark.driver.host", "master") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8,com.mysql:mysql-connector-j:8.0.33") \
    .config("spark.sql.streaming.checkpointLocation", "hdfs://master:9000/DACK/checkpoint") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("1. Tai Hop den va Mo hinh...")
pipeline_model = PipelineModel.load("hdfs://master:9000/DACK/weather_pipeline_model")
lr_model = LogisticRegressionModel.load("hdfs://master:9000/DACK/lr_model_best")

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("Date", StringType(), True), StructField("Location", StringType(), True),
    StructField("MinTemp", DoubleType(), True), StructField("MaxTemp", DoubleType(), True),
    StructField("Rainfall", DoubleType(), True), StructField("WindGustDir", StringType(), True),
    StructField("WindGustSpeed", IntegerType(), True), StructField("WindDir9am", StringType(), True),
    StructField("WindDir3pm", StringType(), True), StructField("WindSpeed9am", IntegerType(), True),
    StructField("WindSpeed3pm", IntegerType(), True), StructField("Humidity9am", IntegerType(), True),
    StructField("Humidity3pm", IntegerType(), True), StructField("Pressure9am", DoubleType(), True),
    StructField("Pressure3pm", DoubleType(), True), StructField("Temp3pm", DoubleType(), True),
    StructField("RainToday", StringType(), True)
])

print("2. Ket noi Kafka...")
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "26.49.14.99:9092") \
    .option("subscribe", "db-mariadb-weather_raw") \
    .option("startingOffsets", "earliest") \
    .load()

# --- JSON VÀ CHUYỂN TIẾP RAW_VALUE ---
parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as raw_value") \
    .withColumn("actual_payload", coalesce(
    get_json_object(col("raw_value"), "$.payload.after"),
    get_json_object(col("raw_value"), "$.payload"),
    get_json_object(col("raw_value"), "$.after"),
    col("raw_value")
)) \
    .withColumn("data", from_json(col("actual_payload"), schema)) \
    .select("raw_value", "data.*")

# --- FEATURE ENGINEERIN ---
stream_fe_df = parsed_df.withColumn("TempRange", col("MaxTemp") - col("MinTemp")) \
    .withColumn("HumidityDiff", col("Humidity9am") - col("Humidity3pm")) \
    .withColumn("PressureDiff", col("Pressure9am") - col("Pressure3pm")) \
    .withColumn("WindSpeedDiff", col("WindSpeed3pm") - col("WindSpeed9am"))

date_formats = ["yyyy-MM-dd", "dd/MM/yyyy", "M/d/yyyy", "yyyy/MM/dd", "d/M/yyyy"]
date_exprs = [to_date(col("Date"), fmt) for fmt in date_formats]

stream_fe_df = stream_fe_df.withColumn("DateParsed", coalesce(*date_exprs)) \
    .withColumn("Month", month(col("DateParsed")))

valid_directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
stream_fe_df = stream_fe_df \
    .withColumn("WindGustDir", when(col("WindGustDir").isin(valid_directions), col("WindGustDir")).otherwise("N")) \
    .withColumn("WindDir9am", when(col("WindDir9am").isin(valid_directions), col("WindDir9am")).otherwise("N")) \
    .withColumn("WindDir3pm", when(col("WindDir3pm").isin(valid_directions), col("WindDir3pm")).otherwise("N")) \
    .withColumn("RainToday", when(col("RainToday").isin(["Yes", "No"]), col("RainToday")).otherwise("No"))

print("3. He thong bat dau du doan...")


# --- HÀM XỬ LÝ LÔ ---
def process_batch(batch_df, batch_id):
    print(f"\n========== LO DU LIEU SO {batch_id} ==========")
    if batch_df.isEmpty():
        print("-> Lo trong (0 dòng). Dang cho du lieu...")
        return
    valid_df = batch_df.dropna(subset=["Location", "MinTemp", "Month"])
    if valid_df.isEmpty():
        print("CANH BAO: Du lieu bi loai vì loi giai ma JSON hoac sai dinh dang Date!")
        return
    try:
        features_df = pipeline_model.transform(valid_df)
        predictions_df = lr_model.transform(features_df)
        columns_to_select = schema.fieldNames() + ["prediction"]
        final_df = predictions_df.select(*columns_to_select)
        if final_df.isEmpty():
            print("MO HINH DANH ROT DU LIEU VI CO COT NULL!")
            return
        print("3. KET QUA DU DOAN THANH CONG:")
        final_df.show(10, False)
        jdbc_url = "jdbc:mysql://localhost:3306/weather_db"
        db_properties = {"user": "root", "password": "123456", "driver": "com.mysql.cj.jdbc.Driver"}
        final_df.write.jdbc(url=jdbc_url, table="weather_predictions", mode="append", properties=db_properties)
        print(f"--> DA LUU THANH CONG VAO MARIADB!")
    except Exception as e:
        print(f" LOI TRONG QUA TRINH DU DOAN: {e}")
query = stream_fe_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", "hdfs://master:9000/DACK/checkpoint") \
    .start()
query.awaitTermination()