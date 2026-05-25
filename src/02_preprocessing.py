# ================================================================
# FILE: 02_preprocessing.py — Tiền xử lý dữ liệu
# ================================================================
# Mục đích:
#   Dựa trên kết quả EDA, làm sạch dữ liệu và tạo ra 3 file:
#     1. weather_clean    → Dùng cho Spark SQL (giữ nguyên chữ)
#     2. weather_ml_rain  → Dùng cho bài toán dự đoán mưa
#     3. weather_ml_fire  → Dùng cho bài toán cháy rừng (có nhãn tổng hợp)
#
# Thứ tự chạy: Sau 01_eda.py, trước 03_spark_sql.py
# ================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from data_loader import get_spark, load_raw_csv
import config

# ----------------------------------------------------------------
# KHỞI TẠO SPARK
# ----------------------------------------------------------------
spark = get_spark("02_Preprocessing_WeatherAUS")


# ================================================================
# BƯỚC 1: ĐỌC DATA GỐC
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 1: ĐỌC DATA GỐC TỪ HDFS")
print("=" * 60)

df = load_raw_csv(spark)

from pyspark.sql.functions import when, col
from config import NUM_COLS

# Thay "NA" thành null thật sự, rồi cast sang kiểu số
for c in NUM_COLS:
    df = df.withColumn(
        c,
        when(col(c) == "NA", None)
        .otherwise(col(c).cast("double"))
    )

# --- Thêm đoạn này sau khi load data ---
from pyspark.sql.types import DoubleType

print("\n--- Đang ép kiểu các cột số sang Double ---")
for col_name in config.NUM_COLS:
    df = df.withColumn(col_name, df[col_name].cast(DoubleType()))
print("✅ Đã ép kiểu xong các cột số!")

# ⭐ ĐIỂM CỘNG +3: persist() vì bước này dùng df nhiều lần
# để tính median, mode cho từng cột
df.persist()

print(f"✅ Data gốc: {df.count():,} dòng × {len(df.columns)} cột")


# ================================================================
# BƯỚC 2: CHUẨN HÓA CỘT DATE
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 2: CHUẨN HÓA CỘT DATE")
print("=" * 60)
# "01/12/2008" (chuỗi) → 2008-12-01 (kiểu Date thực sự)
# Cần chuyển để sau này tính Year/Month/Quarter được chính xác
df = df.withColumn(
    "Date",
    F.to_date(F.col("Date"), "dd/MM/yyyy")
)
print("✅ Đã chuyển cột Date sang kiểu DateType chuẩn ISO")


# ================================================================
# BƯỚC 3: ĐIỀN MISSING VALUES
# Chiến lược từ EDA:
#   - Cột số (NUM_COLS): điền bằng MEDIAN — ít bị ảnh hưởng bởi outliers
#     (không dùng mean vì Rainfall có max=371mm làm lệch trung bình)
#   - Cột chữ (CAT_COLS): điền bằng MODE (giá trị xuất hiện nhiều nhất)
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 3: ĐIỀN MISSING VALUES")
print("=" * 60)
print("Chiến lược: Số → Median | Chữ → Mode\n")

# --- Cột số: điền median ---
print("Đang tính median từng cột số...")
for col_name in config.NUM_COLS:
    # approxQuantile: tính gần đúng nhanh hơn tính chính xác
    # [0.5] = phân vị thứ 50% = median
    # 0.01 = sai số tối đa 1% — chấp nhận được
    median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
    df = df.fillna({col_name: median_val})
    print(f"  {col_name:<22} → điền {median_val:.2f}")

# --- Cột chữ: điền mode ---
print("\nĐang tính mode từng cột chữ...")
for col_name in config.CAT_COLS:
    # Tìm giá trị xuất hiện nhiều nhất
    mode_val = (
        df.groupBy(col_name)
          .count()
          .orderBy(F.desc("count"))
          .first()[0]   # Lấy giá trị cột đầu tiên của dòng đầu tiên
    )
    df = df.fillna({col_name: mode_val})
    print(f"  {col_name:<22} → điền '{mode_val}'")

# --- Xóa các dòng vẫn còn thiếu ở cột quan trọng ---
before = df.count()
# RainTomorrow là target chính — dòng nào thiếu thì không dùng được
df = df.dropna(subset=["RainTomorrow", "Location"])
after = df.count()
print(f"\n✅ Xóa {before - after:,} dòng thiếu cột target")
print(f"✅ Còn lại: {after:,} dòng")


# ================================================================
# BƯỚC 4: THÊM CỘT THỜI GIAN (dùng chung cho cả 3 nhánh)
# ================================================================
print("\n" + "=" * 60)
print("  BƯỚC 4: THÊM CỘT THỜI GIAN")
print("=" * 60)

df = df \
    .withColumn("Year",    F.year("Date")) \
    .withColumn("Month",   F.month("Date")) \
    .withColumn("Quarter", F.quarter("Date"))

# Thêm tên mùa để tiện phân tích (Nam bán cầu — Úc)
# Mùa hè Úc: tháng 12, 1, 2 (ngược với Bắc bán cầu)
df = df.withColumn(
    "Season",
    F.when(F.col("Month").isin([12, 1, 2]),  "Summer")  # Mùa hè
     .when(F.col("Month").isin([3, 4, 5]),   "Autumn")  # Mùa thu
     .when(F.col("Month").isin([6, 7, 8]),   "Winter")  # Mùa đông
     .otherwise("Spring")                               # Mùa xuân
)

print("✅ Đã thêm: Year, Month, Quarter, Season")
df.select("Date", "Year", "Month", "Quarter", "Season").show(5)


# ================================================================
# NHÁNH 1: WEATHER_CLEAN — dùng cho Spark SQL
# Giữ nguyên chữ (Yes/No, Sydney...) vì SQL đọc được chữ
# và kết quả truy vấn dễ đọc hơn khi còn chữ
# ================================================================
print("\n" + "=" * 60)
print("  NHÁNH 1: TẠO weather_clean (cho Spark SQL)")
print("=" * 60)

df_clean = df  # Giữ nguyên, không encoding gì thêm

# Lưu dạng Parquet — nhanh hơn CSV nhiều lần khi Spark đọc
# vì Parquet là định dạng cột (columnar), chỉ đọc cột cần thiết
df_clean.write.mode("overwrite").parquet(config.CLEAN_DATA_PATH)
print(f"✅ Đã lưu weather_clean → {config.CLEAN_DATA_PATH}")
print(f"   {df_clean.count():,} dòng × {len(df_clean.columns)} cột")


# ================================================================
# NHÁNH 2: WEATHER_ML_RAIN — dùng cho bài toán dự đoán mưa
# Giữ chữ vì Pipeline MLlib sẽ tự encoding (StringIndexer)
# ================================================================
print("\n" + "=" * 60)
print("  NHÁNH 2: TẠO weather_ml_rain (cho bài toán mưa)")
print("=" * 60)

df_rain = df.dropna(subset=["RainTomorrow"])

df_rain.write.mode("overwrite").parquet(config.ML_RAIN_PATH)
print(f"✅ Đã lưu weather_ml_rain → {config.ML_RAIN_PATH}")
print(f"   {df_rain.count():,} dòng × {len(df_rain.columns)} cột")

# In phân phối target để xác nhận imbalanced
print("\nPhân phối RainTomorrow trong tập ML:")
df_rain.groupBy("RainTomorrow").count() \
       .withColumn("Pct", F.round(F.col("count") / df_rain.count() * 100, 1)) \
       .show()


# ================================================================
# NHÁNH 3: WEATHER_ML_FIRE — dùng cho bài toán cháy rừng
#
# Kỹ thuật đặc biệt: SYNTHETIC LABELING
# Vì dataset không có cột "nguy cơ cháy rừng" sẵn,
# ta TẠO NHÃN từ Business Rules dựa trên kiến thức chuyên ngành.
# ================================================================
print("\n" + "=" * 60)
print("  NHÁNH 3: TẠO weather_ml_fire (cho bài toán cháy rừng)")
print("  Kỹ thuật: SYNTHETIC LABELING + DATA LEAKAGE PREVENTION")
print("=" * 60)

# --- Bước 3a: Tạo nhãn Bushfire_Risk_Level ---
print("\nBước 3a: Tạo nhãn Bushfire_Risk_Level từ Business Rules...")
print("  Class 3 (Extreme): MaxTemp>=35 VÀ Humidity3pm<=20 VÀ WindGustSpeed>=50")
print("  Class 2 (High)   : MaxTemp>=30 VÀ Humidity3pm<=30")
print("  Class 1 (Moderate): MaxTemp>=25 VÀ Humidity3pm<=45")
print("  Class 0 (Low)    : Còn lại\n")

df_fire = df.withColumn(
    "Bushfire_Risk_Level",
    F.when(
        # Class 3 — Extreme: Phải thỏa CẢ 3 điều kiện
        (F.col("MaxTemp")       >= 35) &
        (F.col("Humidity3pm")   <= 20) &
        (F.col("WindGustSpeed") >= 50),
        3
    ).when(
        # Class 2 — High: Nóng + khô
        (F.col("MaxTemp")     >= 30) &
        (F.col("Humidity3pm") <= 30),
        2
    ).when(
        # Class 1 — Moderate: Ấm + độ ẩm trung bình
        (F.col("MaxTemp")     >= 25) &
        (F.col("Humidity3pm") <= 45),
        1
    ).otherwise(0)  # Class 0 — Low: tất cả trường hợp còn lại
)

# Kiểm tra phân phối nhãn vừa tạo
total_fire = df_fire.count()
print("Phân phối nhãn Bushfire_Risk_Level:")
df_fire.groupBy("Bushfire_Risk_Level") \
       .count() \
       .orderBy("Bushfire_Risk_Level") \
       .withColumn("Pct", F.round(F.col("count") / total_fire * 100, 2)) \
       .withColumn("Nhan",
           F.when(F.col("Bushfire_Risk_Level") == 0, "Low")
            .when(F.col("Bushfire_Risk_Level") == 1, "Moderate")
            .when(F.col("Bushfire_Risk_Level") == 2, "High")
            .otherwise("Extreme")
       ).show()

# --- Bước 3b: DROP 3 cột tạo nhãn (CHỐNG DATA LEAKAGE) ---
# Nếu KHÔNG drop: model thấy MaxTemp=36, Humidity3pm=15 → tự suy ra label=3
# → Accuracy 100% giả tạo, model không học được gì thực sự
# Sau khi drop: model phải học từ MinTemp, Sunshine, Pressure, Cloud...
# → Mới thực sự "học" mẫu khí hậu
print(f"\nBước 3b: DROP các cột tạo nhãn để chống Data Leakage:")
print(f"  Xóa: {config.FIRE_LEAKAGE_COLS}")
df_fire = df_fire.drop(*config.FIRE_LEAKAGE_COLS)
print(f"  Còn lại {len(df_fire.columns)} cột sau khi drop")

df_fire.write.mode("overwrite").parquet(config.ML_FIRE_PATH)
print(f"\n✅ Đã lưu weather_ml_fire → {config.ML_FIRE_PATH}")
print(f"   {df_fire.count():,} dòng × {len(df_fire.columns)} cột")

# Xác nhận 3 cột đã bị xóa thành công
print("\nCác cột hiện có (xác nhận đã drop):")
print(df_fire.columns)


# ================================================================
# KẾT THÚC
# ================================================================
df.unpersist()
spark.stop()

print("""
✅ Tiền xử lý hoàn tất! Đã tạo 3 file trên HDFS:
   1. weather_clean   → chạy tiếp: python 03_spark_sql.py
   2. weather_ml_rain → chạy tiếp: python 04_mllib_rain.py
   3. weather_ml_fire → chạy tiếp: python 04_mllib_fire.py
""")