# ================================================================
# FILE: data_loader.py — Khởi tạo Spark và đọc dữ liệu dùng chung
# ================================================================
# Mục đích: Tránh viết lại đoạn tạo SparkSession ở mỗi file.
# Các file khác chỉ cần gọi:
#   from data_loader import get_spark, load_parquet, load_raw_csv
# ================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import config  # Import cấu hình từ config.py


def get_spark(app_name: str) -> SparkSession:
    """
    Tạo hoặc lấy lại SparkSession hiện có.

    Tham số:
        app_name: Tên hiển thị trên Spark Web UI (cổng 4040).
                  Đặt khác nhau cho mỗi file để dễ phân biệt.

    Trả về:
        SparkSession đã cấu hình sẵn, kết nối HDFS.

    Tại sao dùng hàm này thay vì viết thẳng vào từng file:
        - Thay đổi cấu hình Spark chỉ cần sửa 1 chỗ duy nhất
        - Tránh copy-paste 10 dòng config vào 5 file khác nhau
        - Nếu sau này chuyển sang cluster thật, chỉ sửa config.py
    """
    spark = (
        SparkSession.builder
        .appName(app_name)

        # Chỉ định nơi chạy Spark — đọc từ config.py
        # local[*] = dùng tất cả CPU cores trên máy hiện tại
        # ⭐ ĐIỂM CỘNG +3: Nếu dùng Multi-node Cluster,
        #    đổi SPARK_MASTER trong config.py thành spark://IP:7077
        .master(config.SPARK_MASTER)

        # Cấp RAM cho Driver process
        .config("spark.driver.memory", config.SPARK_DRIVER_MEM)

        # Chỉ Spark biết đường dẫn "hdfs://" nghĩa là đọc từ Hadoop
        # Không có dòng này, Spark đọc file từ ổ cứng thông thường
        .config("spark.hadoop.fs.defaultFS", config.HDFS_HOST)

        # Tắt cảnh báo thừa khi chạy trên môi trường local
        .config("spark.sql.adaptive.enabled", "true")

        # Cho phép ghi đè file khi lưu kết quả (overwrite)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")

        # getOrCreate: lấy session cũ nếu đã có, tạo mới nếu chưa có
        # Tránh lỗi "SparkContext already exists"
        .getOrCreate()
    )

    # Chỉ in cảnh báo quan trọng, bỏ qua log thừa của Hadoop/Spark
    spark.sparkContext.setLogLevel("WARN")

    return spark


def load_raw_csv(spark: SparkSession):
    """
    Đọc file CSV gốc từ HDFS.

    Trả về DataFrame chưa xử lý gì — dùng trong EDA.
    """
    df = spark.read.csv(
        config.RAW_DATA_PATH,
        header=True,       # Dòng đầu là tên cột
        inferSchema=True   # Spark tự đoán kiểu dữ liệu từng cột
    )
    print(f"[data_loader] Đã đọc CSV gốc: {df.count():,} dòng × {len(df.columns)} cột")
    return df


def load_parquet(spark: SparkSession, path: str):
    """
    Đọc file Parquet từ HDFS.

    Parquet là định dạng lưu trữ cột (columnar) — nhanh hơn CSV
    nhiều lần khi Spark đọc vì chỉ đọc những cột cần thiết.

    Tham số:
        path: Đường dẫn HDFS (lấy từ config.py)
    """
    df = spark.read.parquet(path)
    print(f"[data_loader] Đã đọc Parquet từ {path}")
    print(f"              {df.count():,} dòng × {len(df.columns)} cột")
    return df