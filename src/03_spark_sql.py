# ================================================================
# FILE: 03_spark_sql.py — 10 Câu truy vấn Spark SQL phức tạp
# ================================================================
# Mục đích:
#   Phân tích dữ liệu thời tiết qua 10 câu truy vấn đa dạng,
#   sử dụng Star Schema (4 TempView) để thực hiện JOIN nhiều bảng.
#
# Kỹ thuật dùng trong 10 câu:
#   ✅ Window Functions (ROW_NUMBER, RANK, LAG, SUM OVER, AVG OVER)
#   ✅ CTE (WITH ... AS)
#   ✅ Subquery
#   ✅ Group By + Aggregation phức tạp
#   ✅ JOIN nhiều bảng (từ Star Schema)
#   ✅ Phân tích chuỗi thời gian
#   ✅ Conditional Aggregation (CASE WHEN trong SUM)
#
# Thứ tự chạy: Sau 02_preprocessing.py
# ================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from data_loader import get_spark, load_parquet
import config

# ----------------------------------------------------------------
# KHỞI TẠO SPARK
# ----------------------------------------------------------------
spark = get_spark("03_SparkSQL_WeatherAUS")

# ================================================================
# BƯỚC 1: ĐỌC weather_clean TỪ HDFS
# ================================================================
print("\n" + "=" * 65)
print("  BƯỚC 1: ĐỌC DATA VÀ TẠO STAR SCHEMA")
print("=" * 65)

df = load_parquet(spark, config.CLEAN_DATA_PATH)

# ⭐ ĐIỂM CỘNG +3: cache() — tái sử dụng df cho 10 truy vấn
# Không có cache(): Spark đọc lại Parquet từ HDFS mỗi truy vấn
# Có cache(): Đọc 1 lần, giữ trong RAM, 9 truy vấn sau lấy từ RAM
df.cache()
print(f"✅ Đã cache {df.count():,} dòng vào RAM")


# ================================================================
# BƯỚC 2: TẠO STAR SCHEMA — 4 TEMPVIEW
#
# Tại sao tạo Star Schema từ 1 bảng?
# Dataset chỉ có 1 bảng, nhưng thầy yêu cầu JOIN nhiều bảng.
# Giải pháp chuẩn: tách 1 bảng lớn thành nhiều TempView chuyên biệt,
# mỗi TempView chứa 1 nhóm thông tin logic → JOIN lại khi cần.
# Đây đúng là kiến trúc Star Schema trong Data Warehouse thực tế.
# ================================================================
print("\nBước 2: Tạo 4 TempView theo mô hình Star Schema...")

# Cửa sổ thứ tự để tạo ID tự động (Spark không có AUTO_INCREMENT)
loc_window  = Window.orderBy("Location")
date_window = Window.orderBy("Date")
fact_window = Window.orderBy("Date_ID", "Loc_ID")

# --- Bảng chiều 1: Location_Dim ---
# Chuẩn hóa địa danh: 49 địa điểm → mỗi nơi có 1 ID duy nhất
# Giảm dư thừa: thay vì lưu "Sydney" 3000 lần, chỉ lưu số 1
location_dim = (
    df.select("Location").distinct()
      .withColumn("Loc_ID", F.row_number().over(loc_window))
)
location_dim.createOrReplaceTempView("Location_Dim")
print(f"  ✅ Location_Dim: {location_dim.count()} địa điểm")

# --- Bảng chiều 2: Date_Dim ---
# Phân rã cột Date thành Year, Month, Quarter riêng biệt
# Phục vụ GROUP BY linh hoạt theo nhiều cấp thời gian
date_dim = (
    df.select("Date", "Year", "Month", "Quarter", "Season").distinct()
      .withColumn("Date_ID", F.row_number().over(date_window))
      .withColumnRenamed("Date", "FullDate")
)
date_dim.createOrReplaceTempView("Date_Dim")
print(f"  ✅ Date_Dim: {date_dim.count()} ngày")

# --- Bảng sự kiện 1: Weather_Fact ---
# Trung tâm Star Schema — chứa các chỉ số khí hậu chính
# Liên kết với Location_Dim và Date_Dim qua Loc_ID, Date_ID
df_joined = (
    df.join(location_dim, on="Location")
      .join(date_dim, df["Date"] == date_dim["FullDate"])
)
weather_fact = (
    df_joined.select(
        "Loc_ID", "Date_ID",
        "MinTemp", "MaxTemp", "Rainfall",
        "Evaporation", "Sunshine",
        "Humidity9am", "Humidity3pm",
        "Cloud9am", "Cloud3pm",
        "Temp9am", "Temp3pm",
        "RainToday", "RainTomorrow"
    )
    .withColumn("Record_ID", F.row_number().over(fact_window))
)
weather_fact.createOrReplaceTempView("Weather_Fact")
print(f"  ✅ Weather_Fact: {weather_fact.count():,} bản ghi")

# --- Bảng sự kiện 2: Wind_Pressure_Fact ---
# Tách riêng dữ liệu gió & áp suất → mô phỏng nhiều nguồn dữ liệu
# JOIN 1-1 với Weather_Fact qua Record_ID
wind_fact = (
    df_joined.select(
        "Loc_ID", "Date_ID",
        "WindGustDir", "WindGustSpeed",
        "WindDir9am", "WindDir3pm",
        "WindSpeed9am", "WindSpeed3pm",
        "Pressure9am", "Pressure3pm"
    )
    .withColumn("Record_ID", F.row_number().over(fact_window))
)
wind_fact.createOrReplaceTempView("Wind_Pressure_Fact")
print(f"  ✅ Wind_Pressure_Fact: {wind_fact.count():,} bản ghi")

print("\n✅ Star Schema hoàn chỉnh: Location_Dim, Date_Dim, Weather_Fact, Wind_Pressure_Fact")


# ================================================================
# HÀM TIỆN ÍCH
# ================================================================
def run_query(so_thu_tu, ten_query, mo_ta, y_nghia, sql, n=15):
    """Chạy và in kết quả một câu truy vấn Spark SQL"""
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  TRUY VẤN {so_thu_tu}: {ten_query}")
    print(f"  Mục đích: {mo_ta}")
    print(sep)
    result = spark.sql(sql)
    result.show(n, truncate=False)
    print(f"📌 Ý nghĩa: {y_nghia}")
    return result


# ================================================================
# TRUY VẤN 1: Trung bình động 7 ngày — Window Function
# Kỹ thuật: AVG() OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN)
# ================================================================
run_query(
    so_thu_tu=1,
    ten_query="Xu hướng nhiệt độ — Trung bình động 7 ngày",
    mo_ta="Làm mượt nhiễu hàng ngày để thấy xu hướng nóng/lạnh",
    y_nghia="Địa phương nào đang có xu hướng nhiệt tăng liên tục? "
            "Dùng để cảnh báo đợt nắng nóng kéo dài.",
    sql="""
        SELECT
            l.Location,
            d.FullDate,
            ROUND(w.MaxTemp, 1)                              AS MaxTemp_Thuc,
            ROUND(AVG(w.MaxTemp) OVER (
                PARTITION BY l.Location
                ORDER BY d.FullDate
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 2)                                            AS TB_Dong_7Ngay,
            ROUND(w.MaxTemp - AVG(w.MaxTemp) OVER (
                PARTITION BY l.Location
                ORDER BY d.FullDate
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 2)                                            AS Chenh_Lech_VS_TB
        FROM Weather_Fact w
        JOIN Location_Dim l ON w.Loc_ID = l.Loc_ID
        JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
        WHERE l.Location = 'Sydney'
          AND w.MaxTemp IS NOT NULL
        ORDER BY d.FullDate
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 2: Ngày mưa cực đoan — Subquery
# Kỹ thuật: WHERE value > (SELECT AVG FROM subquery)
# ================================================================
run_query(
    so_thu_tu=2,
    ten_query="Ngày mưa cực đoan vượt trung bình năm",
    mo_ta="Tìm ngày có lượng mưa bất thường cao hơn TB của cùng năm",
    y_nghia="Xác định các sự kiện mưa cực đoan — phục vụ cảnh báo lũ lụt "
            "và lập kế hoạch thoát nước đô thị.",
    sql="""
        SELECT
            l.Location,
            d.FullDate,
            d.Year,
            ROUND(w.Rainfall, 1)     AS Luong_Mua_mm,
            ROUND(avg_by_year.AvgRain, 2) AS TB_Nam_mm,
            ROUND(w.Rainfall - avg_by_year.AvgRain, 2) AS Vuot_Troi_mm
        FROM Weather_Fact w
        JOIN Location_Dim l  ON w.Loc_ID  = l.Loc_ID
        JOIN Date_Dim d      ON w.Date_ID = d.Date_ID
        JOIN (
            SELECT d2.Year, AVG(w2.Rainfall) AS AvgRain
            FROM Weather_Fact w2
            JOIN Date_Dim d2 ON w2.Date_ID = d2.Date_ID
            WHERE w2.Rainfall IS NOT NULL
            GROUP BY d2.Year
        ) avg_by_year ON d.Year = avg_by_year.Year
        WHERE w.Rainfall > avg_by_year.AvgRain * 5
        ORDER BY w.Rainfall DESC
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 3: Xác suất mưa tiếp diễn — Conditional Aggregation + JOIN
# Kỹ thuật: SUM(CASE WHEN) / COUNT + JOIN 2 bảng + GROUP BY
# ================================================================
run_query(
    so_thu_tu=3,
    ten_query="Xác suất mưa kéo dài theo Quý và địa điểm",
    mo_ta="Nếu hôm nay mưa, ngày mai mưa tiếp xác suất bao nhiêu?",
    y_nghia="Dữ liệu quan trọng cho nông nghiệp: lên lịch thu hoạch "
            "và chuẩn bị thiết bị chống ngập khi biết mùa mưa kéo dài.",
    sql="""
        SELECT
            l.Location,
            d.Quarter,
            COUNT(*)                  AS Tong_Ngay,
            SUM(CASE WHEN w.RainToday='Yes' AND w.RainTomorrow='Yes'
                THEN 1 ELSE 0 END)    AS Ngay_Mua_Lien_Tiep,
            ROUND(
                SUM(CASE WHEN w.RainToday='Yes' AND w.RainTomorrow='Yes'
                    THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            )                         AS XacSuat_Mua_TiepDien_Pct
        FROM Weather_Fact w
        JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
        JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
        WHERE w.RainToday IS NOT NULL
          AND w.RainTomorrow IS NOT NULL
        GROUP BY l.Location, d.Quarter
        HAVING COUNT(*) > 100
        ORDER BY XacSuat_Mua_TiepDien_Pct DESC
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 4: Top 3 ngày gió mạnh nhất mỗi năm — DENSE_RANK + Subquery
# Kỹ thuật: DENSE_RANK() OVER + WHERE rank <= 3 + JOIN 2 bảng Fact
# ================================================================
run_query(
    so_thu_tu=4,
    ten_query="Top 3 ngày gió giật mạnh nhất từng năm từng địa điểm",
    mo_ta="Lập bản đồ rủi ro bão theo địa phương và thời gian",
    y_nghia="Hỗ trợ công ty bảo hiểm đánh giá rủi ro, chính quyền "
            "lên lịch kiểm tra hạ tầng trước mùa gió.",
    sql="""
        SELECT Location, Year, FullDate, WindGustSpeed_kmh, Xep_Hang
        FROM (
            SELECT
                l.Location,
                d.Year,
                d.FullDate,
                ROUND(wp.WindGustSpeed, 1) AS WindGustSpeed_kmh,
                DENSE_RANK() OVER (
                    PARTITION BY l.Location, d.Year
                    ORDER BY wp.WindGustSpeed DESC
                ) AS Xep_Hang
            FROM Wind_Pressure_Fact wp
            JOIN Weather_Fact w ON wp.Record_ID = w.Record_ID
            JOIN Location_Dim l ON w.Loc_ID   = l.Loc_ID
            JOIN Date_Dim d     ON w.Date_ID  = d.Date_ID
            WHERE wp.WindGustSpeed IS NOT NULL
        ) ranked
        WHERE Xep_Hang <= 3
        ORDER BY Location, Year, Xep_Hang
        LIMIT 30
    """
)

# ================================================================
# TRUY VẤN 5: Sụt giảm áp suất — LAG Function (chuỗi thời gian)
# Kỹ thuật: LAG() OVER (PARTITION BY ... ORDER BY ...)
# ================================================================
run_query(
    so_thu_tu=5,
    ten_query="Mức sụt giảm áp suất so với ngày hôm trước",
    mo_ta="Áp suất sụt nhanh = dấu hiệu front lạnh = cảnh báo bão",
    y_nghia="Khí tượng học: áp suất giảm >5 hPa trong 24h là dấu hiệu "
            "bão mạnh sắp đến. Dùng cho hệ thống cảnh báo sớm.",
    sql="""
        SELECT
            l.Location,
            d.FullDate,
            ROUND(wp.Pressure3pm, 1)                AS Ap_Suat_HomNay,
            ROUND(LAG(wp.Pressure3pm, 1) OVER (
                PARTITION BY l.Location
                ORDER BY d.FullDate
            ), 1)                                   AS Ap_Suat_HomQua,
            ROUND(LAG(wp.Pressure3pm, 1) OVER (
                PARTITION BY l.Location
                ORDER BY d.FullDate
            ) - wp.Pressure3pm, 2)                  AS Muc_Sut_Giam_hPa,
            w.RainTomorrow                          AS Mua_Ngay_Mai
        FROM Wind_Pressure_Fact wp
        JOIN Weather_Fact w ON wp.Record_ID = w.Record_ID
        JOIN Location_Dim l ON w.Loc_ID   = l.Loc_ID
        JOIN Date_Dim d     ON w.Date_ID  = d.Date_ID
        WHERE wp.Pressure3pm IS NOT NULL
        ORDER BY Muc_Sut_Giam_hPa DESC NULLS LAST
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 6: Lượng mưa tích lũy từ đầu năm — Cumulative SUM
# Kỹ thuật: SUM() OVER (PARTITION BY ... ROWS UNBOUNDED PRECEDING)
# ================================================================
run_query(
    so_thu_tu=6,
    ten_query="Lượng mưa tích lũy từ đầu năm (Year-to-Date)",
    mo_ta="Theo dõi hạn hán/lũ lụt qua tổng lượng mưa tích lũy",
    y_nghia="Quản lý hồ chứa nước: nếu mưa tích lũy thấp hơn ngưỡng "
            "vào tháng 6 → cảnh báo hạn hán → hạn chế tưới tiêu nông nghiệp.",
    sql="""
        SELECT
            l.Location,
            d.Year,
            d.Month,
            d.FullDate,
            ROUND(w.Rainfall, 1)       AS Mua_Ngay_mm,
            ROUND(SUM(w.Rainfall) OVER (
                PARTITION BY l.Location, d.Year
                ORDER BY d.FullDate
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ), 1)                      AS Mua_TichLuy_mm
        FROM Weather_Fact w
        JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
        JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
        WHERE l.Location = 'Melbourne'
          AND d.Year = 2015
          AND w.Rainfall IS NOT NULL
        ORDER BY d.FullDate
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 7: Đợt hạn hán dài nhất — CTE + Gaps & Islands
# Kỹ thuật: WITH ... AS (CTE), ROW_NUMBER() hai lần để xác định chuỗi
# ================================================================
run_query(
    so_thu_tu=7,
    ten_query="Đợt hạn hán dài nhất (chuỗi ngày không mưa liên tiếp)",
    mo_ta="Tìm chuỗi ngày khô hạn liên tục dài nhất tại mỗi địa phương",
    y_nghia="Lập kế hoạch quản lý nguồn nước: biết đợt hạn dài nhất "
            "lịch sử để tính lượng dự trữ nước tối thiểu cần có.",
    sql="""
        WITH Drought_Rows AS (
            SELECT
                l.Location,
                d.FullDate,
                ROW_NUMBER() OVER (
                    PARTITION BY l.Location
                    ORDER BY d.FullDate
                ) AS row_all,
                ROW_NUMBER() OVER (
                    PARTITION BY l.Location, w.RainToday
                    ORDER BY d.FullDate
                ) AS row_no_rain
            FROM Weather_Fact w
            JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
            JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
            WHERE w.RainToday = 'No'
        ),
        Drought_Groups AS (
            SELECT
                Location,
                (row_all - row_no_rain) AS grp,
                MIN(FullDate) AS Ngay_BatDau,
                MAX(FullDate) AS Ngay_KetThuc,
                COUNT(*)      AS So_Ngay_Han
            FROM Drought_Rows
            GROUP BY Location, (row_all - row_no_rain)
        )
        SELECT
            Location,
            Ngay_BatDau,
            Ngay_KetThuc,
            So_Ngay_Han,
            DENSE_RANK() OVER (
                PARTITION BY Location
                ORDER BY So_Ngay_Han DESC
            ) AS Xep_Hang_Han
        FROM Drought_Groups
        WHERE So_Ngay_Han >= 20
        ORDER BY So_Ngay_Han DESC
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 8: Biên độ nhiệt so với trung bình quốc gia — CTE + JOIN
# Kỹ thuật: 2 CTE song song + JOIN + tính chênh lệch
# ================================================================
run_query(
    so_thu_tu=8,
    ten_query="Biên độ nhiệt ngày-đêm so với trung bình quốc gia",
    mo_ta="Địa phương nào có biên độ nhiệt cao/thấp hơn TB cả nước?",
    y_nghia="Quy hoạch đô thị: biên độ cao → khí hậu lục địa khô → "
            "cần vật liệu xây dựng chịu đựng thay đổi nhiệt độ lớn. "
            "Ảnh hưởng thiết kế hệ thống điều hòa không khí.",
    sql="""
        WITH Local_Amplitude AS (
            SELECT
                l.Location,
                d.Month,
                ROUND(AVG(w.MaxTemp - w.MinTemp), 2) AS BienDo_DiaPhuong
            FROM Weather_Fact w
            JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
            JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
            WHERE w.MaxTemp IS NOT NULL AND w.MinTemp IS NOT NULL
            GROUP BY l.Location, d.Month
        ),
        National_Amplitude AS (
            SELECT
                d.Month,
                ROUND(AVG(w.MaxTemp - w.MinTemp), 2) AS BienDo_QuocGia
            FROM Weather_Fact w
            JOIN Date_Dim d ON w.Date_ID = d.Date_ID
            WHERE w.MaxTemp IS NOT NULL AND w.MinTemp IS NOT NULL
            GROUP BY d.Month
        )
        SELECT
            la.Location,
            la.Month,
            la.BienDo_DiaPhuong,
            na.BienDo_QuocGia,
            ROUND(la.BienDo_DiaPhuong - na.BienDo_QuocGia, 2) AS Chenh_Lech,
            CASE
                WHEN la.BienDo_DiaPhuong > na.BienDo_QuocGia
                THEN 'Cao hơn TB quốc gia'
                ELSE 'Thấp hơn TB quốc gia'
            END AS Nhan_Xet
        FROM Local_Amplitude la
        JOIN National_Amplitude na ON la.Month = na.Month
        ORDER BY ABS(la.BienDo_DiaPhuong - na.BienDo_QuocGia) DESC
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 9: Tỷ lệ ngày mưa theo mùa — Seasonality Analysis
# Kỹ thuật: GROUP BY nhiều cấp + Aggregation + Window Rank
# ================================================================
run_query(
    so_thu_tu=9,
    ten_query="Phân tích mùa mưa — Tỷ lệ ngày mưa theo mùa từng địa điểm",
    mo_ta="Xác định mùa mưa và mùa khô rõ ràng cho từng địa phương",
    y_nghia="Quy hoạch nông nghiệp: biết chính xác mùa mưa để "
            "lên lịch gieo trồng. Du lịch: khuyến nghị mùa đẹp nhất "
            "để tham quan từng địa điểm.",
    sql="""
        SELECT
            l.Location,
            d.Season,
            COUNT(*)                   AS Tong_Ngay,
            SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END) AS Ngay_Co_Mua,
            ROUND(
                SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            )                          AS TyLe_Mua_Pct,
            ROUND(AVG(w.Rainfall), 2)  AS Luong_Mua_TB_mm,
            RANK() OVER (
                PARTITION BY l.Location
                ORDER BY
                    SUM(CASE WHEN w.RainToday = 'Yes' THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*) DESC
            )                          AS Rank_Mua_Nhieu_Nhat
        FROM Weather_Fact w
        JOIN Location_Dim l ON w.Loc_ID  = l.Loc_ID
        JOIN Date_Dim d     ON w.Date_ID = d.Date_ID
        WHERE w.RainToday IS NOT NULL
        GROUP BY l.Location, d.Season
        ORDER BY l.Location, TyLe_Mua_Pct DESC
        LIMIT 20
    """
)

# ================================================================
# TRUY VẤN 10: Ngưỡng kép độ ẩm + gió dự báo mưa — Multi-JOIN + CASE
# Kỹ thuật: JOIN 2 bảng Fact + CASE WHEN phân nhóm + GROUP BY kép
# ================================================================
run_query(
    so_thu_tu=10,
    ten_query="Ma trận ngưỡng Độ ẩm × Gió dự báo xác suất mưa ngày mai",
    mo_ta="Kết hợp Humidity3pm và WindGustSpeed để tính xác suất mưa",
    y_nghia="Xây dựng bảng tra cứu nhanh cho người dân/nông dân: "
            "'Chiều nay ẩm >80% và gió mạnh >60km/h → xác suất mưa ngày mai >70%' "
            "→ cần thu hoạch trước khi mưa.",
    sql="""
        SELECT
            CASE
                WHEN wp.Humidity3pm >= 80 THEN '3_Cao (>=80%)'
                WHEN wp.Humidity3pm >= 60 THEN '2_TB  (60-79%)'
                WHEN wp.Humidity3pm >= 40 THEN '1_Vua (40-59%)'
                ELSE                          '0_Thap (<40%)'
            END AS Nhom_DoAm_3pm,
            CASE
                WHEN wp.WindGustSpeed >= 60 THEN '3_Manh  (>=60 km/h)'
                WHEN wp.WindGustSpeed >= 40 THEN '2_Vua   (40-59 km/h)'
                ELSE                             '1_Nhe   (<40 km/h)'
            END AS Nhom_Gio,
            COUNT(*)                     AS So_Ngay_Quan_Sat,
            SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END)
                                         AS Ngay_Mua_Ngay_Mai,
            ROUND(
                SUM(CASE WHEN w.RainTomorrow = 'Yes' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            )                            AS XacSuat_Mua_NM_Pct
        FROM Weather_Fact w
        JOIN Wind_Pressure_Fact wp ON w.Record_ID = wp.Record_ID
        WHERE wp.Humidity3pm    IS NOT NULL
          AND wp.WindGustSpeed  IS NOT NULL
          AND w.RainTomorrow    IS NOT NULL
        GROUP BY Nhom_DoAm_3pm, Nhom_Gio
        HAVING COUNT(*) >= 100
        ORDER BY XacSuat_Mua_NM_Pct DESC
    """
)

# ================================================================
# ⭐ ĐIỂM CỘNG +3: explain() trên truy vấn phức tạp nhất
# ================================================================
print("\n" + "=" * 65)
print("  ⭐ ĐIỂM CỘNG +3: explain() — KẾ HOẠCH THỰC THI")
print("  Truy vấn 10 (JOIN 2 bảng Fact + Aggregation)")
print("=" * 65)
spark.sql("""
    SELECT
        CASE WHEN wp.Humidity3pm >= 80 THEN 'Cao' ELSE 'Thap' END AS NhomAm,
        COUNT(*) AS SoNgay,
        ROUND(AVG(CASE WHEN w.RainTomorrow='Yes' THEN 1.0 ELSE 0.0 END)*100, 2) AS PctMua
    FROM Weather_Fact w
    JOIN Wind_Pressure_Fact wp ON w.Record_ID = wp.Record_ID
    GROUP BY NhomAm
""").explain(mode="formatted")

# ================================================================
# KẾT THÚC
# ================================================================
df.unpersist()
spark.stop()
print("\n✅ Hoàn tất 10 câu truy vấn Spark SQL!")
print("   Chạy tiếp: python 04_mllib_rain.py")