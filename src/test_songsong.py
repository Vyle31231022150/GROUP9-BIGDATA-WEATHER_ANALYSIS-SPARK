from pyspark.sql import SparkSession
import socket
import time

spark = SparkSession.builder \
    .appName("KiemTraSongSong") \
    .config("spark.network.timeout", "1000s") \
    .config("spark.executor.heartbeatInterval", "100s") \
    .getOrCreate()

# Đọc file dữ liệu từ HDFS
df = spark.read.csv("hdfs://master:9000/test/weatherAUS.csv", header=True)

# Hàm dò tên máy tính đang trực tiếp xử lý dữ liệu
def do_ten_may(iterator):
    ten_may = socket.gethostname()
    yield ten_may

# ĐƯỜNG DẪN LƯU KẾT QUẢ
output_path = "hdfs://master:9000/test/ket_qua_phan_tan"

# XÓA THƯ MỤC CŨ (Để không bị lỗi FileAlreadyExists)
# Lệnh này dùng để xóa kết quả cũ trong HDFS trước khi chạy
import subprocess
# Hoặc sửa dòng 25 thành thế này:
subprocess.run(["hadoop", "fs", "-rm", "-r", "-f", "/test/ket_qua_phan_tan"], shell=True)
# GIAO VIỆC VÀ LƯU KẾT QUẢ TRỰC TIẾP LÊN HDFS
# Việc này làm hoàn toàn song song, không cần kéo về Driver nên sẽ không bao giờ bị lỗi mạng
df.rdd.mapPartitions(do_ten_may).distinct().saveAsTextFile(output_path)

print("="*50)
print(f"ĐÃ GHI KẾT QUẢ VÀO HDFS TẠI ĐƯỜNG DẪN: {output_path}")
print("BẠN CÓ THỂ MỞ TRANG HADOOP UI ĐỂ XEM HOẶC DÙNG LỆNH HADOOP ĐỂ ĐỌC FILE")
print("="*50)

time.sleep(120)
spark.stop()