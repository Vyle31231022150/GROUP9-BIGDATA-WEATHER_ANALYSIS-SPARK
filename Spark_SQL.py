# ==============================================================================
# PHÂN TÍCH DỮ LIỆU THỜI TIẾT NƯỚC ÚC (SPARK SQL)
#
# Mô hình Galaxy Schema gồm 4 TempView:
# - Location_Dim: Chiều không gian (49 địa điểm)
# - Date_Dim: Chiều thời gian (Year / Month / Quarter / Season)
# - Weather_Fact: Sự kiện khí hậu (nhiệt độ, độ ẩm, mưa)
# - Wind_Pressure_Fact: Động lực học không khí (gió, áp suất)
# ==============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# BƯỚC 1 — KHỞI TẠO SPARK SESSION VÀ ĐỌC DỮ LIỆU
# ==============================================================================
spark = (
    SparkSession.builder
    .appName("WeatherAUS_SparkSQL_10Queries")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")  # Đổi thành ERROR để tắt sạch rác log màu đỏ
print("✅ SparkSession khởi động thành công!\n")

print("Đang đọc file CSV đã qua tiền xử lý từ HDFS...")
df = spark.read.csv(
    "hdfs://master:9000/DACK/weather_clean",
    header=True,
    inferSchema=True
)
print(f"Đã đọc: {df.count():,} dòng × {len(df.columns)} cột\n")

# ==============================================================================
# BƯỚC 2 — XÂY DỰNG GALAXY SCHEMA (4 TEMP VIEW)
# ==============================================================================
# ── Chuẩn bị cột thời gian ──
df = (
    df
    .withColumn("ParsedDate", F.to_date("DateParsed"))
    .withColumn("Quarter", F.quarter("ParsedDate"))
    .withColumn("Season",
                F.when(F.col("Month").isin([12, 1, 2]), "Summer")
                .when(F.col("Month").isin([3, 4, 5]), "Autumn")
                .when(F.col("Month").isin([6, 7, 8]), "Winter")
                .otherwise("Spring")
                )
)

# ── Tạo ID tự động bằng ROW_NUMBER ──
w_loc = Window.orderBy("Location")
w_date = Window.orderBy("ParsedDate")
w_fact = Window.orderBy("ParsedDate", "Location")

# BẢNG 1: Location_Dim (Chiều không gian)
location_dim = df.select("Location").distinct().withColumn("Loc_ID", F.row_number().over(w_loc))
location_dim.createOrReplaceTempView("Location_Dim")

# BẢNG 2: Date_Dim (Chiều thời gian)
date_dim = (
    df.select("ParsedDate", "Year", "Month", "Quarter", "Season")
    .distinct()
    .withColumn("Date_ID", F.row_number().over(w_date))
    .withColumnRenamed("ParsedDate", "FullDate")
)
date_dim.createOrReplaceTempView("Date_Dim")

# JOIN df với 2 bảng chiều để lấy ID
df_j = (
    df
    .join(location_dim, on="Location")
    .join(
        date_dim.withColumnRenamed("FullDate", "_fd"),
        df["ParsedDate"] == F.col("_fd"),
        "left"
    )
    .withColumn("Record_ID", F.row_number().over(w_fact))
)

# BẢNG 3: Weather_Fact (Sự kiện khí hậu chính)
weather_fact = df_j.select(
    "Record_ID", "Loc_ID", "Date_ID", "MinTemp", "MaxTemp", "Rainfall",
    "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow"
)
weather_fact.createOrReplaceTempView("Weather_Fact")

# BẢNG 4: Wind_Pressure_Fact (Động lực học không khí)
wind_pressure_fact = df_j.select(
    "Record_ID", "Loc_ID", "Date_ID", "WindGustDir", "WindGustSpeed",
    "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Pressure3pm"
)
wind_pressure_fact.createOrReplaceTempView("Wind_Pressure_Fact")

print("Đã tạo 4 TempView thành công:")
print(f"   Location_Dim      : {location_dim.count()} địa điểm")
print(f"   Date_Dim          : {date_dim.count():,} ngày")
print(f"   Weather_Fact      : {weather_fact.count():,} bản ghi")
print(f"   Wind_Pressure_Fact: {wind_pressure_fact.count():,} bản ghi\n")


# ==============================================================================
# HÀM CHẠY SQL IN RA BẢNG KHUNG VIỀN ĐẸP
# ==============================================================================
def run_query(sql, title=""):
    if title:
        print(f"\n{'=' * 100}")
        print(f" {title} ")
        print(f"{'=' * 100}")

    result = spark.sql(sql)

    # Lấy dữ liệu (Mở rộng cho phép lấy tối đa 1000 dòng để luôn hiển thị hết kết quả)
    pdf = result.limit(1000).toPandas()

    # Vẽ bảng cực đẹp, canh lề siêu thẳng bằng tabulate (tablefmt='psql')
    print(tabulate(pdf, headers='keys', tablefmt='psql', showindex=False))

    print(f"\n   → Tổng số dòng kết quả: {result.count():,}\n")
    return result


# ==============================================================================
# THỰC THI 10 CÂU TRUY VẤN
# ==============================================================================

# ──────────────────────────────────────────────────────────────
# CÂU 1: Phát hiện năm El Niño (hạn hán) và La Niña (lũ lụt)
# ──────────────────────────────────────────────────────────────
q1 = run_query("""
    WITH MuaTheoNam AS (
        SELECT
            d.Year,
            COUNT(*) AS Tong_Ngay,
            SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END) AS So_Ngay_Mua,
            ROUND(
                SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS TyLe_Mua_Pct,
            ROUND(AVG(w.Rainfall), 2) AS LuongMua_TB_mm
        FROM Weather_Fact w
        JOIN Date_Dim d ON w.Date_ID = d.Date_ID
        GROUP BY d.Year
    ),
    TrungBinhToanKy AS (
        SELECT AVG(LuongMua_TB_mm) AS TB_ToanKy
        FROM MuaTheoNam
    )
    SELECT
        m.Year, m.Tong_Ngay, m.So_Ngay_Mua, m.TyLe_Mua_Pct, m.LuongMua_TB_mm,
        ROUND(t.TB_ToanKy, 2) AS TB_ToanKy_mm,
        ROUND(m.LuongMua_TB_mm - t.TB_ToanKy, 2) AS Lech_So_TB,
        CASE
            WHEN m.LuongMua_TB_mm > t.TB_ToanKy * 1.15 THEN '🌊 La Niña — Lũ lụt'
            WHEN m.LuongMua_TB_mm < t.TB_ToanKy * 0.85 THEN '🔥 El Niño — Hạn hán'
            ELSE 'Bình thường'
        END AS Ket_Luan_Khi_Hau
    FROM MuaTheoNam m, TrungBinhToanKy t
    ORDER BY m.Year
""", title="CÂU 1: PHÁT HIỆN NĂM EL NIÑO & LA NIÑA (2007-2017)")

# ──────────────────────────────────────────────────────────────
# CÂU 2: Mùa mưa và mùa khô của 4 vùng khí hậu
# ──────────────────────────────────────────────────────────────
q2 = run_query("""
    SELECT
        l.Location, d.Season, COUNT(*) AS Tong_Ngay,
        SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END) AS So_Ngay_Mua,
        ROUND(SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS TyLe_Mua_Pct,
        ROUND(AVG(w.Rainfall), 2) AS LuongMua_TB_mm,
        ROUND(AVG(w.Humidity3pm), 1) AS DoAm_3pm_TB,
        RANK() OVER (
            PARTITION BY l.Location
            ORDER BY SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) DESC
        ) AS Rank_Mua_NhieuNhat
    FROM Weather_Fact w
    JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
    JOIN Date_Dim d ON w.Date_ID = d.Date_ID
    WHERE l.Location IN ('Darwin', 'Sydney', 'Melbourne', 'Cobar') AND w.RainToday IS NOT NULL
    GROUP BY l.Location, d.Season
    ORDER BY l.Location, TyLe_Mua_Pct DESC
""", title="CÂU 2: MÙA MƯA VÀ MÙA KHÔ CỦA 4 VÙNG KHÍ HẬU ÚC")

# ──────────────────────────────────────────────────────────────
# CÂU 3: Áp suất giảm mạnh báo hiệu mưa ngày mai
# ──────────────────────────────────────────────────────────────
q3 = run_query("""
    WITH ApSuatVaSutGiam AS (
        SELECT
            l.Location, d.FullDate, wp.Pressure3pm,
            LAG(wp.Pressure3pm, 1) OVER (PARTITION BY l.Location ORDER BY d.FullDate) AS ApSuat_HomQua,
            LAG(wp.Pressure3pm, 1) OVER (PARTITION BY l.Location ORDER BY d.FullDate) - wp.Pressure3pm AS Sut_Giam_hPa,
            w.RainTomorrow
        FROM Wind_Pressure_Fact wp
        JOIN Weather_Fact w ON wp.Record_ID = w.Record_ID
        JOIN Location_Dim l ON w.Loc_ID     = l.Loc_ID
        JOIN Date_Dim d     ON w.Date_ID    = d.Date_ID
        WHERE wp.Pressure3pm IS NOT NULL AND w.RainTomorrow IS NOT NULL
    )
    SELECT
        CASE
            WHEN Sut_Giam_hPa > 5  THEN '4_Giảm mạnh  (> 5 hPa)'
            WHEN Sut_Giam_hPa > 2  THEN '3_Giảm vừa  (2–5 hPa)'
            WHEN Sut_Giam_hPa >= 0 THEN '2_Ổn định   (0–2 hPa)'
            ELSE                        '1_Tăng      (< 0 hPa)'
        END AS Muc_SutGiam_ApSuat,
        COUNT(*) AS So_Ngay_Quan_Sat,
        SUM(CASE WHEN RainTomorrow = 'Yes' THEN 1 ELSE 0 END) AS Ngay_Mua_NM,
        ROUND(SUM(CASE WHEN RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS XacSuat_Mua_NM_Pct
    FROM ApSuatVaSutGiam
    WHERE Sut_Giam_hPa IS NOT NULL
    GROUP BY Muc_SutGiam_ApSuat
    ORDER BY Muc_SutGiam_ApSuat DESC
""", title="CÂU 3: MỨC SỤT GIẢM ÁP SUẤT VÀ XÁC SUẤT MƯA NGÀY MAI")

# ──────────────────────────────────────────────────────────────
# CÂU 4: Hướng gió báo hiệu mưa theo vùng
# ──────────────────────────────────────────────────────────────
q4 = run_query("""
    SELECT
        l.Location, wp.WindGustDir AS Huong_Gio_Giat, COUNT(*) AS So_Ngay_Quan_Sat,
        SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) AS Ngay_Mua_NM,
        ROUND(SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS XacSuat_Mua_NM_Pct,
        ROUND(AVG(wp.WindGustSpeed), 1) AS TocDo_Gio_TB_kmh
    FROM Wind_Pressure_Fact wp
    JOIN Weather_Fact w ON wp.Record_ID = w.Record_ID
    JOIN Location_Dim l ON w.Loc_ID     = l.Loc_ID
    WHERE l.Location IN ('Darwin', 'Sydney', 'Melbourne', 'Cobar')
      AND wp.WindGustDir IS NOT NULL AND w.RainTomorrow IS NOT NULL
    GROUP BY l.Location, wp.WindGustDir
    HAVING COUNT(*) >= 50
    ORDER BY l.Location, XacSuat_Mua_NM_Pct DESC
""", title="CÂU 4: HƯỚNG GIÓ BÁO HIỆU MƯA CAO NHẤT THEO VÙNG")

# ──────────────────────────────────────────────────────────────
# CÂU 5: Ngưỡng độ ẩm chiều báo mưa
# ──────────────────────────────────────────────────────────────
q5 = run_query("""
    WITH PhanViDoAm AS (
        SELECT
            l.Location, w.Humidity3pm, w.RainTomorrow,
            NTILE(10) OVER (PARTITION BY l.Location ORDER BY w.Humidity3pm) AS Nhom_DoAm
        FROM Weather_Fact w
        JOIN Location_Dim l ON w.Loc_ID = l.Loc_ID
        WHERE w.Humidity3pm IS NOT NULL AND w.RainTomorrow IS NOT NULL
    )
    SELECT
        Location, Nhom_DoAm,
        ROUND(MIN(Humidity3pm), 0) AS DoAm_Min_Pct,
        ROUND(MAX(Humidity3pm), 0) AS DoAm_Max_Pct,
        COUNT(*) AS So_Ngay,
        ROUND(SUM(CASE WHEN RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS XacSuat_Mua_NM_Pct,
        CASE
            WHEN SUM(CASE WHEN RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) >= 50 THEN '⚠️ Vượt ngưỡng 50%'
            ELSE ''
        END AS Canh_Bao
    FROM PhanViDoAm
    WHERE Location IN ('Darwin', 'Sydney', 'Melbourne', 'Cobar')
    GROUP BY Location, Nhom_DoAm
    ORDER BY Location, Nhom_DoAm
""", title="CÂU 5: NGƯỠNG ĐỘ ẨM CHIỀU XÁC SUẤT MƯA VƯỢT 50%")

# ──────────────────────────────────────────────────────────────
# CÂU 6: Ma trận xác suất mưa (Độ ẩm × Áp suất)
# ──────────────────────────────────────────────────────────────
q6 = run_query("""
    SELECT
        CASE
            WHEN w.Humidity3pm >= 80 THEN '3_Cao  (>= 80%)'
            WHEN w.Humidity3pm >= 50 THEN '2_TB   (50–79%)'
            ELSE                          '1_Thấp (< 50%)'
        END AS Nhom_DoAm_3pm,
        CASE
            WHEN wp.Pressure3pm < 1011 THEN '1_Thấp  (< 1011 hPa)'
            WHEN wp.Pressure3pm < 1019 THEN '2_TB    (1011–1019)'
            ELSE                            '3_Cao   (>= 1019 hPa)'
        END AS Nhom_ApSuat_3pm,
        COUNT(*) AS So_Ngay_Quan_Sat,
        SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) AS Ngay_Mua_NM,
        ROUND(SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS XacSuat_Mua_NM_Pct
    FROM Weather_Fact w
    JOIN Wind_Pressure_Fact wp ON w.Record_ID = wp.Record_ID
    WHERE w.Humidity3pm IS NOT NULL AND wp.Pressure3pm IS NOT NULL AND w.RainTomorrow IS NOT NULL
    GROUP BY Nhom_DoAm_3pm, Nhom_ApSuat_3pm
    HAVING COUNT(*) >= 500
    ORDER BY XacSuat_Mua_NM_Pct DESC
""", title="CÂU 6: MA TRẬN XÁC SUẤT MƯA KHI KẾT HỢP ĐỘ ẨM VÀ ÁP SUẤT")

# ──────────────────────────────────────────────────────────────
# CÂU 7: Lượng mưa tích lũy (La Niña vs El Niño)
# ──────────────────────────────────────────────────────────────
q7 = run_query("""
    SELECT
        d.Year, d.FullDate, d.Month,
        ROUND(w.Rainfall, 1) AS Mua_Ngay_mm,
        ROUND(
            SUM(w.Rainfall) OVER (
                PARTITION BY l.Location, d.Year
                ORDER BY d.FullDate
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ), 1
        ) AS Mua_TichLuy_YTD_mm,
        CASE d.Year
            WHEN 2010 THEN '🌊 La Niña 2010 (lũ lụt)'
            WHEN 2014 THEN '🔥 El Niño 2014 (hạn hán)'
        END AS Loai_Nam
    FROM Weather_Fact w
    JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
    JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
    WHERE l.Location = 'Melbourne' AND d.Year IN (2010, 2014)
    ORDER BY d.Year, d.FullDate
""", title="CÂU 7: ĐƯỜNG MƯA TÍCH LŨY (LA NINA 2010 VS EL NINO 2014 TẠI MELBOURNE)")

# ==============================================================================
# TRỰC QUAN HÓA CÂU 7: ĐƯỜNG MƯA TÍCH LŨY
# ==============================================================================
print("Đang vẽ biểu đồ Câu 7...")

# 1. Chuyển kết quả Spark DataFrame của câu 7 thành Pandas DataFrame
pdf7 = q7.toPandas()

# 2. Xử lý thời gian để đè 2 đường lên cùng 1 trục X
# Ép kiểu cột FullDate về định dạng datetime
pdf7['FullDate'] = pd.to_datetime(pdf7['FullDate'])

# Tạo cột 'DayOfYear' (Ngày thứ 1 đến ngày 365) để 2 năm 2010 và 2014 có thể nằm đè lên nhau
pdf7['DayOfYear'] = pdf7['FullDate'].dt.dayofyear

# 3. Cài đặt khung vẽ
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# 4. Vẽ biểu đồ đường (Lineplot)
sns.lineplot(
    data=pdf7,
    x='DayOfYear',
    y='Mua_TichLuy_YTD_mm',
    hue='Loai_Nam',          # Tự động chia 2 màu cho 2 năm
    palette=["#e74c3c", "#3498db"], # Màu: Đỏ (El Niño) - Xanh (La Niña)
    linewidth=3
)

# 5. Trang trí tiêu đề, nhãn dán
plt.title('So sánh Đường Mưa Tích Lũy: La Niña (2010) vs El Niño (2014) tại Melbourne', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Ngày trong năm (Từ 1/1 đến 31/12)', fontsize=12)
plt.ylabel('Lượng mưa tích lũy (mm)', fontsize=12)
plt.legend(title='Chu kỳ khí hậu', fontsize=11)

# Chỉ định mốc các tháng trên trục X cho dễ nhìn (30 ngày, 60 ngày...)
plt.xticks(ticks=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
           labels=['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
                   'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12'], rotation=45)

# 6. Hiển thị biểu đồ (Code sẽ tạm dừng ở đây cho đến khi bạn đóng cửa sổ biểu đồ)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────
# CÂU 8: Đợt khô hạn liên tiếp dài nhất lịch sử
# ──────────────────────────────────────────────────────────────
q8 = run_query("""
    WITH ChuoiNgayKho AS (
        SELECT
            l.Location, d.FullDate,
            ROW_NUMBER() OVER (PARTITION BY l.Location ORDER BY d.FullDate) AS row_tat_ca,
            ROW_NUMBER() OVER (PARTITION BY l.Location, w.RainToday ORDER BY d.FullDate) AS row_khong_mua
        FROM Weather_Fact w
        JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
        JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
        WHERE w.RainToday = 'No'
    ),
    NhomHanHan AS (
        SELECT
            Location,
            (row_tat_ca - row_khong_mua) AS nhom_chuoi,
            MIN(FullDate) AS Ngay_BatDau,
            MAX(FullDate) AS Ngay_KetThuc,
            COUNT(*)      AS So_Ngay_Han
        FROM ChuoiNgayKho
        GROUP BY Location, (row_tat_ca - row_khong_mua)
    )
    SELECT
        Location, Ngay_BatDau, Ngay_KetThuc, So_Ngay_Han,
        DENSE_RANK() OVER (ORDER BY So_Ngay_Han DESC) AS Xep_Hang_Toan_Quoc
    FROM NhomHanHan
    WHERE So_Ngay_Han >= 50
    ORDER BY So_Ngay_Han DESC
    LIMIT 15
""", title="CÂU 8: ĐỢT KHÔ HẠN LIÊN TIẾP DÀI NHẤT TẠI MỖI ĐỊA ĐIỂM")

# ──────────────────────────────────────────────────────────────
# CÂU 9: Xác suất mưa kéo dài sang ngày hôm sau
# ──────────────────────────────────────────────────────────────
q9 = run_query("""
    SELECT
        l.Location, d.Season, COUNT(*) AS Tong_Ngay,
        SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END) AS Ngay_Mua_HomNay,
        SUM(CASE WHEN w.RainToday = 'Yes' AND w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) AS Mua_Ca_Hai_Ngay,
        ROUND(
            SUM(CASE WHEN w.RainToday = 'Yes' AND w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0
            / NULLIF(SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END), 0), 1
        ) AS XacSuat_MuaTiepDien_Pct
    FROM Weather_Fact w
    JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
    JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
    WHERE w.RainToday IS NOT NULL AND w.RainTomorrow IS NOT NULL
      AND l.Location IN ('Darwin', 'Sydney', 'Melbourne', 'Cobar')
    GROUP BY l.Location, d.Season
    HAVING COUNT(*) >= 100
    ORDER BY l.Location, XacSuat_MuaTiepDien_Pct DESC
""", title="CÂU 9: XÁC SUẤT MƯA KÉO DÀI (HÔM NAY MƯA -> MAI MƯA TIẾP)")

# ──────────────────────────────────────────────────────────────
# CÂU 10: Xếp hạng tổng hợp 49 địa điểm
# ──────────────────────────────────────────────────────────────
q10 = run_query("""
    SELECT
        l.Location, COUNT(*) AS Tong_BanGhi,
        ROUND(AVG(w.MaxTemp), 1) AS NhietDo_Max_TB,
        ROUND(AVG(w.MinTemp), 1) AS NhietDo_Min_TB,
        ROUND(AVG(w.Humidity3pm), 1) AS DoAm_3pm_TB,
        ROUND(AVG(wp.Pressure3pm), 1) AS ApSuat_3pm_TB,
        ROUND(AVG(wp.WindGustSpeed), 1) AS TocDoGio_TB_kmh,
        ROUND(AVG(w.Rainfall), 2) AS LuongMua_TB_mm,
        ROUND(SUM(w.Rainfall), 0) AS LuongMua_Tong_10Nam_mm,
        ROUND(SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS TyLe_Mua_NM_Pct,
        RANK() OVER (
            ORDER BY SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) DESC
        ) AS Rank_Mua_NhieuNhat
    FROM Weather_Fact w
    JOIN Wind_Pressure_Fact wp ON w.Record_ID = wp.Record_ID
    JOIN Location_Dim l        ON w.Loc_ID    = l.Loc_ID
    JOIN Date_Dim d            ON w.Date_ID   = d.Date_ID
    GROUP BY l.Location
    ORDER BY TyLe_Mua_NM_Pct DESC
""", title="CÂU 10: BẢNG TỔNG HỢP XẾP HẠNG 49 ĐỊA ĐIỂM TRONG 10 NĂM")

# ==============================================================================
# KẾT THÚC
# ==============================================================================
spark.stop()
print("✅ Đã đóng SparkSession. Hoàn tất toàn bộ kịch bản 10 câu truy vấn!")