# ================================================================
# FILE: config.py — Cấu hình trung tâm toàn bộ project
# ================================================================
# Mục đích: Tập trung tất cả đường dẫn, tham số vào 1 file duy nhất.
# Khi cần thay đổi (ví dụ đổi IP Hadoop), chỉ sửa ở đây,
# tất cả các file khác tự động cập nhật theo → tránh lặp code.
# ================================================================

# ---------------------------------------------------------------
# CẤU HÌNH HADOOP HDFS
# ---------------------------------------------------------------
# Địa chỉ NameNode của Hadoop — nơi lưu trữ dữ liệu phân tán
# localhost:9000 là cấu hình mặc định khi chạy Single-node
#
# ⭐ ĐIỂM CỘNG +3 (Multi-node Cluster):
# Nếu đã cài Multi-node Cluster, đổi dòng dưới thành:
# HDFS_HOST = "hdfs://192.168.1.100:9000"  ← IP máy Master thật
HDFS_HOST = "hdfs://localhost:9000"

# Thư mục gốc lưu toàn bộ dữ liệu project trên HDFS
HDFS_BASE = f"{HDFS_HOST}/weather"

# ---------------------------------------------------------------
# ĐƯỜNG DẪN FILE DỮ LIỆU TRÊN HDFS
# ---------------------------------------------------------------
# File gốc — upload lên HDFS trước khi chạy bất kỳ script nào
RAW_DATA_PATH     = f"{HDFS_BASE}/weatherAUS.csv"

# File sau tiền xử lý — được tạo ra bởi 02_preprocessing.py
CLEAN_DATA_PATH   = f"{HDFS_BASE}/weather_clean"      # Dùng cho Spark SQL
ML_RAIN_PATH      = f"{HDFS_BASE}/weather_ml_rain"    # Dùng cho bài toán mưa
ML_FIRE_PATH      = f"{HDFS_BASE}/weather_ml_fire"    # Dùng cho bài toán cháy rừng

# Thư mục lưu model sau khi train
MODEL_RAIN_PATH   = f"{HDFS_BASE}/model_rain"
MODEL_FIRE_PATH   = f"{HDFS_BASE}/model_fire"

# Thư mục giả lập streaming (trên máy local, không phải HDFS)
STREAM_INPUT_DIR  = "/tmp/weather_stream_input"
STREAM_OUTPUT_DIR = "/tmp/weather_stream_output"

# ---------------------------------------------------------------
# CẤU HÌNH SPARK
# ---------------------------------------------------------------
# "local[*]" = dùng tất cả CPU cores trên máy hiện tại
#
# ⭐ ĐIỂM CỘNG +3 (Multi-node Cluster):
# Nếu đã cài cluster, đổi thành:
# SPARK_MASTER = "spark://192.168.1.100:7077"
SPARK_MASTER      = "local[*]"

# RAM cấp cho Spark Driver (tiến trình điều phối chính)
# 4g = 4GB — phù hợp với dataset 145K dòng
SPARK_DRIVER_MEM  = "8g"

# ---------------------------------------------------------------
# CẤU HÌNH CÁC CỘT DỮ LIỆU
# ---------------------------------------------------------------

# Cột số trong dataset gốc
NUM_COLS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm"
]

# Cột phân loại (dạng chữ) trong dataset gốc
CAT_COLS = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

# Cột target bài toán mưa
TARGET_RAIN = "RainTomorrow"

# Cột target bài toán cháy rừng (sẽ được tạo ra trong preprocessing)
TARGET_FIRE = "Bushfire_Risk_Level"

# ---------------------------------------------------------------
# CẤU HÌNH BÀI TOÁN CHÁY RỪNG
# ---------------------------------------------------------------
# 3 cột dùng để TẠO nhãn → phải DROP khỏi tập train (chống Data Leakage)
FIRE_LEAKAGE_COLS = ["MaxTemp", "Humidity3pm", "WindGustSpeed"]

# Features dùng để train bài toán cháy rừng
# (sau khi đã loại bỏ FIRE_LEAKAGE_COLS)
FIRE_NUM_FEATURES = [
    "MinTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am",                    # Humidity3pm đã bị drop
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm",             # MaxTemp đã bị drop
    "Year", "Month", "Quarter"        # Thời gian — quan trọng cho cháy rừng mùa hè
]
FIRE_CAT_FEATURES = ["WindGustDir", "WindDir9am", "WindDir3pm",
                     "RainToday", "Location"]

# Features dùng để train bài toán mưa
RAIN_NUM_FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm"
]
RAIN_CAT_FEATURES = ["WindGustDir", "WindDir9am", "WindDir3pm",
                     "RainToday", "Location"]

# ---------------------------------------------------------------
# NHÃN CHÁY RỪNG — dùng để in kết quả dễ đọc
# ---------------------------------------------------------------
FIRE_RISK_LABELS = {
    0: "Low (Thấp)",
    1: "Moderate (Trung bình)",
    2: "High (Cao)",
    3: "Extreme (Cực độ)"
}