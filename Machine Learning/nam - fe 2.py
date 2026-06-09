from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler

# ==========================================
# PHẦN 1: KHỞI TẠO VÀ ĐỌC DỮ LIỆU SẠCH V2
# ==========================================
spark = SparkSession.builder \
    .appName("Weather_Feature_Engineering") \
    .master("local[*]") \
    .getOrCreate()

print("Đang đọc dữ liệu siêu sạch từ bước tiền xử lý...")
# Đọc thư mục chứa file CSV sạch vừa tạo ở bước trước
df = spark.read.csv("weather_clean_v2", header=True, inferSchema=True)


# ==========================================
# PHẦN 2: TẠO THÊM CÁC ĐẶC TRƯNG MỚI (FEATURE CREATION)
# ==========================================
print("Đang tính toán các chỉ số chênh lệch (Nghiệp vụ khí tượng)...")

# 1. Biên độ nhiệt độ trong ngày (Cao nhất - Thấp nhất)
df = df.withColumn("TempRange", col("MaxTemp") - col("MinTemp"))

# 2. Chênh lệch độ ẩm giữa sáng và chiều (9am - 3pm)
df = df.withColumn("HumidityDiff", col("Humidity9am") - col("Humidity3pm"))

# 3. Chênh lệch áp suất giữa sáng và chiều (9am - 3pm)
df = df.withColumn("PressureDiff", col("Pressure9am") - col("Pressure3pm"))

# 4. Chênh lệch tốc độ gió giữa chiều và sáng (3pm - 9am)
df = df.withColumn("WindSpeedDiff", col("WindSpeed3pm") - col("WindSpeed9am"))


# ==========================================
# PHẦN 3: ĐÓNG GÓI HÀNH LÝ (VECTOR ASSEMBLER)
# ==========================================
print("Đang gom nhóm 22 biến độc lập thành một cột 'features'...")

# Liệt kê đầy đủ danh sách 22 biến độc lập theo đúng logic bảng phân tích
feature_cols = [
    # Nhóm Nhiệt độ
    "MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "TempRange",
    # Nhóm Lượng mưa
    "Rainfall",
    # Nhóm Độ ẩm
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    # Nhóm Áp suất
    "Pressure9am", "Pressure3pm", "PressureDiff",
    # Nhóm Gió
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "WindSpeedDiff",
    # Nhóm Thời gian (Mùa vụ)
    "Month",
    # Nhóm các Biến phân loại đã mã hóa thành số
    "Location_index", "WindGustDir_index", "WindDir9am_index", "WindDir3pm_index", "RainToday_index"
]

# Gọi công cụ đóng gói Vector của Spark
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Ép bảng dữ liệu đi qua bộ đóng gói này
df_vector = assembler.transform(df)


# ==========================================
# PHẦN 4: CHUẨN HÓA DỮ LIỆU (STANDARD SCALER)
# ==========================================
print("Đang chuẩn hóa thang đo về cùng vạch xuất phát (StandardScaler)...")

# Khai báo bộ chuẩn hóa: Nhận đầu vào là 'features' và trả đầu ra là 'scaled_features'
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Cho bộ chuẩn hóa "học" các tham số (như độ lệch chuẩn) trên tập dữ liệu
scaler_model = scaler.fit(df_vector)

# Thực hiện ép các con số về cùng một thang đo chuẩn
df_final = scaler_model.transform(df_vector)


# ==========================================
# PHẦN 5: LƯU THÀNH PHẨM LÊN HDFS (ĐỊNH DẠNG PARQUET)
# ==========================================
print("Đang lưu bộ dữ liệu hoàn hảo lên HDFS...")

# Chỉ lọc lấy 2 cột quan trọng nhất cho khâu Machine Learning để tiết kiệm bộ nhớ HDFS
df_save = df_final.select("scaled_features", "label")

# Đường dẫn chuẩn lên HDFS mà bạn đã cấu hình
hdfs_output_path = "hdfs://localhost:9000/DACK/weather_ml_rain_index"

# Ghi dữ liệu dạng Parquet (Dạng file nén chuyên dụng tối ưu của Big Data)
df_save.coalesce(1) \
       .write \
       .mode("overwrite") \
       .parquet(hdfs_output_path)

print(f"========= THÀNH CÔNG RỰC RỠ =========")
print(f"Dữ liệu Feature Engineering đã nằm an toàn tại: {hdfs_output_path}")