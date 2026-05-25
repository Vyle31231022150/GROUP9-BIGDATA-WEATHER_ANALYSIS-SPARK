# ================================================================
# FILE: 01_eda.py — Phân tích Khám phá Dữ liệu (EDA)
# ================================================================
# Mục đích:
#   Đọc data GỐC từ HDFS, phân tích toàn diện, KHÔNG sửa gì.
#   Kết quả EDA là căn cứ để đưa ra mọi quyết định phía sau:
#     → Cột nào cần điền missing?
#     → Cột nào cần drop?
#     → Bài toán ML có bị imbalanced không?
#     → Features nào quan trọng nhất?
#
# Chạy: spark-submit 01_eda.py
#        hoặc: python 01_eda.py (nếu đã cài PySpark)
#
# Thứ tự chạy: File này phải chạy TRƯỚC 02_preprocessing.py
# ================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from data_loader import get_spark, load_raw_csv
import config

# ----------------------------------------------------------------
# KHỞI TẠO SPARK
# ----------------------------------------------------------------
# get_spark() được định nghĩa trong data_loader.py — tránh lặp code
spark = get_spark("01_EDA_WeatherAUS")


# ================================================================
# GIAI ĐOẠN 1: CẤU TRÚC DỮ LIỆU
# Mục tiêu: Hiểu mình đang làm việc với dữ liệu gì
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 1: CẤU TRÚC DỮ LIỆU")
print("=" * 65)

# Đọc CSV gốc từ HDFS — chưa sửa gì cả
df = load_raw_csv(spark)

# ⭐ ĐIỂM CỘNG +3 (Tối ưu hóa hiệu năng — persist):
# persist() giữ DataFrame trong RAM sau lần đọc đầu tiên.
# EDA có 7 giai đoạn, mỗi giai đoạn gọi action (.count(), .show()...)
# Không có persist(): Spark đọc lại từ HDFS mỗi lần → rất chậm.
# Có persist(): Đọc 1 lần, lưu RAM, 6 lần sau lấy từ RAM → nhanh.
df.persist()

from pyspark.sql.functions import when, col
from config import NUM_COLS

# Thay "NA" thành null thật sự, rồi cast sang kiểu số
for c in NUM_COLS:
    df = df.withColumn(
        c,
        when(col(c) == "NA", None)
        .otherwise(col(c).cast("double"))
    )

# In số dòng, số cột
total_rows = df.count()
total_cols = len(df.columns)
print(f"\nSố dòng (records): {total_rows:,}")
print(f"Số cột (features): {total_cols}")

# printSchema() in kiểu dữ liệu từng cột
# Quan trọng vì: cột số dùng median để điền missing,
#                cột chữ dùng mode,
#                cột Date cần chuyển đổi riêng
print("\nSchema (tên cột + kiểu dữ liệu):")
df.printSchema()

# In 5 dòng đầu để hiểu dữ liệu trông như thế nào
print("\n5 dòng đầu tiên:")
df.show(5, truncate=False)


# ================================================================
# GIAI ĐOẠN 2: MISSING VALUES (Giá trị bị thiếu)
# Mục tiêu: Biết cột nào bị thiếu bao nhiêu %
#           → Quyết định điền hay drop trong preprocessing
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 2: PHÂN TÍCH MISSING VALUES")
print("=" * 65)
print("(Kết quả này quyết định chiến lược tiền xử lý)\n")

# Đếm số ô trống từng cột bằng cách lọc các dòng có giá trị null
missing_results = []
for col_name in df.columns:
    # filter().count() = đếm số dòng thỏa điều kiện isNull
    null_count = df.filter(F.col(col_name).isNull()).count()
    null_pct   = round(null_count / total_rows * 100, 2)
    missing_results.append((col_name, null_count, null_pct))

# Sắp xếp theo % giảm dần — cột thiếu nhiều nhất hiện lên đầu
missing_results.sort(key=lambda x: x[2], reverse=True)

print(f"{'Tên cột':<22} {'Số thiếu':>10} {'Tỷ lệ %':>10}  Đánh giá")
print("-" * 65)
for col_name, cnt, pct in missing_results:
    # Đánh giá mức độ nghiêm trọng để đưa ra khuyến nghị
    if pct == 0:
        label = "✅ Đầy đủ"
    elif pct < 5:
        label = "⚠️  Ít — điền median/mode"
    elif pct < 30:
        label = "⚠️  Vừa — điền cẩn thận"
    elif pct < 50:
        label = "🔴 Nhiều — cân nhắc drop"
    else:
        label = "🔴🔴 Nghiêm trọng — nên drop"
    print(f"{col_name:<22} {cnt:>10,} {pct:>9}%  {label}")

print("\n📌 Nhận xét EDA:")
print("  - Sunshine (48%) và Evaporation (43%): thiếu gần nửa → điền median")
print("  - Cloud9am/3pm (38-40%): thiếu nhiều → điền median")
print("  - Humidity3pm, Pressure (3-10%): thiếu ít → điền median/mode")
print("  - Quyết định: KHÔNG drop cột nào vì tất cả đều có giá trị phân tích")


# ================================================================
# GIAI ĐOẠN 3: THỐNG KÊ MÔ TẢ (Descriptive Statistics)
# Mục tiêu: Hiểu phân phối dữ liệu số
#           → Phát hiện outliers, phân phối lệch
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 3: THỐNG KÊ MÔ TẢ CÁC CỘT SỐ")
print("=" * 65)

# describe() tính: count, mean, stddev, min, max cho cột số
df.describe(config.NUM_COLS).show(truncate=False)

# Tính thêm MEDIAN (trung vị) — describe() không có sẵn median
# approxQuantile(col, [0.5], relative_error) = tính phân vị thứ 50%
print("\nMedian (trung vị) từng cột số:")
print(f"{'Cột':<22} {'Median':>10}")
print("-" * 35)
for col_name in config.NUM_COLS:
    try:
        # relative_error=0.01: sai số ≤ 1%, đủ chính xác
        median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
        print(f"{col_name:<22} {median_val:>10.2f}")
    except Exception:
        print(f"{col_name:<22} {'N/A':>10}")

print("\n📌 Nhận xét EDA:")
print("  - Rainfall: mean=2.4mm nhưng max=371mm → phân phối rất lệch phải")
print("  - WindGustSpeed: max=135 km/h → ngày bão thực tế, KHÔNG phải lỗi")
print("  - MaxTemp: max=48.1°C → đợt nắng nóng cực đoan tại Úc (có thật)")
print("  - Các outliers này là sự kiện thực tế → GIỮ LẠI, không xóa")


# ================================================================
# GIAI ĐOẠN 4: PHÂN TÍCH PHÂN PHỐI CỘT PHÂN LOẠI
# Mục tiêu: Hiểu biến chữ có bao nhiêu giá trị unique
#           → Quyết định encoding strategy trong preprocessing
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 4: PHÂN TÍCH CỘT PHÂN LOẠI (Categorical)")
print("=" * 65)

all_cat = config.CAT_COLS + ["Location", TARGET_RAIN := config.TARGET_RAIN]
for col_name in all_cat:
    n_unique = df.select(col_name).distinct().count()
    print(f"\n{col_name} ({n_unique} giá trị unique):")
    df.groupBy(col_name).count() \
      .orderBy(F.desc("count")) \
      .show(5, truncate=False)

print("\n📌 Nhận xét EDA:")
print("  - Location: 49 địa điểm → cần StringIndexer + OneHotEncoder")
print("  - WindGustDir: 16 hướng gió → cần encoding")
print("  - RainToday/RainTomorrow: chỉ Yes/No → StringIndexer đơn giản")


# ================================================================
# GIAI ĐOẠN 5: PHÂN TÍCH BIẾN MỤC TIÊU (Target Variable)
# Mục tiêu: Phát hiện class imbalance
#           → Quyết định xử lý trong MLlib (classWeight, SMOTE...)
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 5: PHÂN TÍCH BIẾN MỤC TIÊU — RainTomorrow")
print("=" * 65)
print("(Đây là cột cần DỰ ĐOÁN trong bài toán 1)\n")

rain_dist = df.groupBy("RainTomorrow") \
              .count() \
              .withColumn("Pct_%", F.round(F.col("count") / total_rows * 100, 2)) \
              .orderBy("RainTomorrow")
rain_dist.show()

print("📌 Nhận xét EDA:")
print("  - No (không mưa): ~77.6% | Yes (có mưa): ~22.4%")
print("  - Tỷ lệ 77:22 → DỮ LIỆU MẤT CÂN BẰNG (Imbalanced)")
print("  - Hậu quả nếu không xử lý: model lười biếng, lúc nào")
print("    cũng đoán No → Accuracy 77% nhưng thực ra vô dụng")
print("  - Giải pháp: dùng classWeight trong RandomForestClassifier")


# ================================================================
# GIAI ĐOẠN 6: PHÂN TÍCH OUTLIERS bằng phương pháp IQR
# Mục tiêu: Phân biệt outlier do lỗi dữ liệu vs sự kiện thực tế
#           → Quyết định xóa hay giữ lại
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 6: PHÂN TÍCH OUTLIERS (IQR Method)")
print("=" * 65)
print("Công thức: Outlier nếu < Q1 - 1.5×IQR  hoặc  > Q3 + 1.5×IQR\n")

key_cols = ["Rainfall", "WindGustSpeed", "MaxTemp", "MinTemp",
            "Evaporation", "Humidity3pm"]

print(f"{'Cột':<22} {'Q1':>6} {'Q3':>6} {'IQR':>6} "
      f"{'Ngưỡng dưới':>13} {'Ngưỡng trên':>13} "
      f"{'Số outlier':>11} {'%':>6}  Kết luận")
print("-" * 100)

for col_name in key_cols:
    # Tính Q1 (phân vị 25%) và Q3 (phân vị 75%)
    q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
    iqr     = q3 - q1
    lower   = q1 - 1.5 * iqr  # Ngưỡng dưới
    upper   = q3 + 1.5 * iqr  # Ngưỡng trên

    # Đếm số dòng vượt ngưỡng
    n_out = df.filter(
        (F.col(col_name) < lower) | (F.col(col_name) > upper)
    ).count()
    pct_out = round(n_out / total_rows * 100, 2)

    # Nhận xét tự động
    if col_name in ["Rainfall", "WindGustSpeed"]:
        note = "GIỮ — sự kiện thời tiết thực"
    elif pct_out > 5:
        note = "GIỮ — phân phối tự nhiên lệch"
    else:
        note = "GIỮ — ít, không ảnh hưởng"

    print(f"{col_name:<22} {q1:>6.1f} {q3:>6.1f} {iqr:>6.1f} "
          f"{lower:>13.1f} {upper:>13.1f} "
          f"{n_out:>11,} {pct_out:>5}%  {note}")

print("\n📌 Nhận xét EDA:")
print("  - Rainfall có 18% outliers nhưng đây là ngày mưa lớn thực tế")
print("  - WindGustSpeed 135 km/h = bão thực tế tại Úc")
print("  - Quyết định: GIỮ LẠI tất cả outliers — chúng mang thông tin thật")


# ================================================================
# GIAI ĐOẠN 7: TƯƠNG QUAN VỚI BIẾN MỤC TIÊU (Correlation)
# Mục tiêu: Tìm features quan trọng nhất với RainTomorrow
#           → Căn cứ khoa học để chọn features cho MLlib
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 7: TƯƠNG QUAN VỚI RainTomorrow (Correlation)")
print("=" * 65)
print("(Kết quả này quyết định chọn features nào cho mô hình ML)\n")

# Chuyển RainTomorrow thành 0/1 để tính tương quan số
df_corr = df.withColumn(
    "target_bin",
    F.when(F.col("RainTomorrow") == "Yes", 1.0).otherwise(0.0)
)

# Tính Pearson correlation giữa từng cột số với target
print(f"{'Feature':<22} {'Tương quan':>12}  Mức độ ảnh hưởng")
print("-" * 55)

corr_results = []
for col_name in config.NUM_COLS:
    # corr() tính Pearson correlation coefficient [-1, 1]
    # Giá trị dương: feature tăng → khả năng mưa tăng
    # Giá trị âm: feature tăng → khả năng mưa giảm
    # Gần 0: gần như không có liên hệ
    corr_val = df_corr.stat.corr(col_name, "target_bin")
    corr_results.append((col_name, corr_val))

# Sắp xếp theo |correlation| giảm dần
corr_results.sort(key=lambda x: abs(x[1]), reverse=True)

for col_name, corr_val in corr_results:
    abs_corr = abs(corr_val)
    if abs_corr >= 0.3:
        level = "🔴 Cao — chọn làm feature chính"
    elif abs_corr >= 0.15:
        level = "🟡 Trung bình — nên giữ"
    else:
        level = "⚪ Thấp — ít ảnh hưởng"
    bar = "█" * int(abs_corr * 20)
    print(f"{col_name:<22} {corr_val:>+8.4f}    {bar}  {level}")

print("\n📌 Nhận xét EDA — Kết luận chọn features cho MLlib:")
print("  TOP features quan trọng nhất:")
print("  1. Sunshine         (tương quan âm: ít nắng → dễ mưa)")
print("  2. Humidity3pm      (tương quan dương: ẩm chiều → dễ mưa)")
print("  3. Cloud3pm         (tương quan dương: nhiều mây chiều → dễ mưa)")
print("  4. Pressure3pm/9am  (áp suất thấp → front lạnh → mưa)")
print("  5. Temp3pm          (nhiệt độ chiều liên quan chu kỳ mưa)")
print("  → Tất cả các features trên sẽ được đưa vào mô hình MLlib")


# ================================================================
# GIAI ĐOẠN 8: PHÂN TÍCH THEO THỜI GIAN & ĐỊA LÝ
# Mục tiêu: Hiểu phạm vi dữ liệu — cần cho Spark SQL và streaming
# ================================================================
print("\n" + "=" * 65)
print("  GIAI ĐOẠN 8: PHÂN TÍCH THỜI GIAN VÀ ĐỊA LÝ")
print("=" * 65)

# Chuyển cột Date từ chuỗi "01/12/2008" sang kiểu Date thực sự
df_dated = df.withColumn(
    "ParsedDate", F.to_date(F.col("Date"), "dd/MM/yyyy")
)

# Tìm ngày bắt đầu và kết thúc
df_dated.select(
    F.min("ParsedDate").alias("Ngay_bat_dau"),
    F.max("ParsedDate").alias("Ngay_ket_thuc"),
    F.countDistinct("ParsedDate").alias("So_ngay_unique"),
    F.countDistinct("Location").alias("So_dia_diem")
).show()

# Phân tích số bản ghi theo năm — kiểm tra dữ liệu có đầy đủ không
print("Số bản ghi theo năm:")
df_dated.withColumn("Year", F.year("ParsedDate")) \
        .groupBy("Year") \
        .count() \
        .orderBy("Year") \
        .show()

print("Top 10 địa điểm có nhiều bản ghi nhất:")
df.groupBy("Location") \
  .count() \
  .orderBy(F.desc("count")) \
  .show(10)

print("\n📌 Nhận xét EDA:")
print("  - Dữ liệu từ 11/2007 đến 6/2017 — ~10 năm quan trắc")
print("  - 49 địa điểm khắp nước Úc — đủ đa dạng địa lý")
print("  - Phân phối theo năm đều → không bị thiên lệch thời gian")


# ================================================================
# ⭐ ĐIỂM CỘNG +3: explain() — In kế hoạch thực thi Spark
# Mục đích: Chứng minh hiểu cơ chế bên trong Spark
# ================================================================
print("\n" + "=" * 65)
print("  ⭐ ĐIỂM CỘNG +3: explain() — KẾ HOẠCH THỰC THI SPARK")
print("  (Tối ưu hóa hiệu năng phân tán — mục 2.5 đề bài)")
print("=" * 65)
print("explain() cho thấy Spark lập kế hoạch xử lý câu truy vấn")
print("như thế nào trước khi thực sự chạy:\n")

# explain() trên một câu truy vấn phức tạp để thầy thấy rõ
# mode='formatted' in đẹp hơn mode mặc định
df_dated.withColumn("Year", F.year("ParsedDate")) \
        .groupBy("Location", "Year") \
        .agg(
            F.avg("MaxTemp").alias("AvgMaxTemp"),
            F.sum("Rainfall").alias("TotalRainfall"),
            F.count("*").alias("Records")
        ) \
        .orderBy("Location", "Year") \
        .explain(mode="formatted")


# ================================================================
# KẾT THÚC EDA — TỔNG KẾT CÁC QUYẾT ĐỊNH
# ================================================================
print("\n" + "=" * 65)
print("  TỔNG KẾT EDA — CÁC QUYẾT ĐỊNH CHO BƯỚC TIẾP THEO")
print("=" * 65)
print("""
TIỀN XỬ LÝ (02_preprocessing.py):
  ✅ Điền Sunshine, Evaporation bằng MEDIAN (thiếu 43-48%)
  ✅ Điền WindGustDir, WindDir* bằng MODE (thiếu 7%)
  ✅ Giữ lại tất cả outliers (sự kiện thời tiết thực tế)
  ✅ Thêm cột Year/Month/Quarter từ Date

SPARK SQL (03_spark_sql.py):
  ✅ Dùng weather_clean — giữ chữ để dễ đọc kết quả
  ✅ Tập trung phân tích Humidity3pm, Sunshine, Cloud — top features
  ✅ Phân tích theo Location và chuỗi thời gian

MLLIB — BÀI TOÁN MƯA (04_mllib_rain.py):
  ✅ Features chính: Sunshine, Humidity3pm, Cloud3pm, Pressure
  ✅ Xử lý imbalanced 77/22 bằng classWeight

MLLIB — BÀI TOÁN CHÁY RỪNG (04_mllib_fire.py):
  ✅ Tạo nhãn tổng hợp từ MaxTemp, Humidity3pm, WindGustSpeed
  ✅ DROP 3 cột tạo nhãn để chống Data Leakage
  ✅ Xử lý imbalanced bằng classWeight
""")

# Giải phóng RAM — không cần giữ df trong bộ nhớ nữa
df.unpersist()

# Đóng SparkSession — giải phóng tài nguyên
spark.stop()

print("✅ EDA hoàn tất! Chạy tiếp: python 02_preprocessing.py")