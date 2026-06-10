from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PHẦN 1: KHỞI TẠO VÀ ĐỌC DỮ LIỆU SẠCH V2
# ==========================================
spark = SparkSession.builder \
    .appName("Weather_Feature_Engineering") \
    .master("local[*]") \
    .getOrCreate()

print("Đang đọc dữ liệu siêu sạch từ bước tiền xử lý...")
df = spark.read.csv("weather_clean_v2", header=True, inferSchema=True)
print("Đang tính toán các chỉ số chênh lệch...")

df = df.withColumn("TempRange", col("MaxTemp") - col("MinTemp"))
df = df.withColumn("HumidityDiff", col("Humidity9am") - col("Humidity3pm"))
df = df.withColumn("PressureDiff", col("Pressure9am") - col("Pressure3pm"))
df = df.withColumn("WindSpeedDiff", col("WindSpeed3pm") - col("WindSpeed9am"))


# ==========================================
# PHẦN 3.5A: KIỂM TRA TƯƠNG QUAN GIỮA FEATURES
# ==========================================

print("Đang kiểm tra tương quan giữa các biến số...")

numeric_cols = [

    "MinTemp",
    "MaxTemp",
    "Temp9am",
    "Temp3pm",
    "TempRange",

    "Rainfall",

    "Humidity9am",
    "Humidity3pm",
    "HumidityDiff",

    "Pressure9am",
    "Pressure3pm",
    "PressureDiff",

    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "WindSpeedDiff",

    "Month"
]

# chuyển sang pandas để corr dễ hơn
pdf = df.select(numeric_cols).toPandas()

corr_matrix = pdf.corr()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# tăng kích thước figure
plt.figure(figsize=(16, 12))

# mask tam giác trên để đỡ rối
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# vẽ heatmap
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,          # hiện số correlation
    fmt=".2f",
    cmap="coolwarm",     # đỏ xanh dễ nhìn
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": .8}
)

plt.title("Feature Correlation Matrix", fontsize=18)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()



print("Đang tính Mutual Information cho các biến có tương quan cao...")

# ==========================================
# Chỉ xét các biến thuộc nhóm corr cao
# ==========================================

high_corr_features = [
    "MinTemp",
    "MaxTemp",
    "Temp9am",
    "Temp3pm",
    "Pressure9am",
    "Pressure3pm"
]

# Spark -> Pandas
pdf = df.select(high_corr_features + ["label"]).toPandas()

# ==========================================
# X và y
# ==========================================

X = pdf.drop(columns=["label"])
y = pdf["label"]

# ==========================================
# Mutual Information
# ==========================================

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

# dataframe kết quả
mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI Score": mi_scores
})

# sort giảm dần
mi_df = mi_df.sort_values(
    by="MI Score",
    ascending=False
)

print("\nMutual Information Ranking:")
print(mi_df)

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=mi_df,
    x="MI Score",
    y="Feature"
)

plt.title("Mutual Information của các biến tương quan cao")
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()





# ==========================================
# PHẦN 2: TẠO THÊM CÁC ĐẶC TRƯNG MỚI
# ==========================================


# ==========================================
# PHẦN 2.5: PEARSON CORRELATION (EDA)
# ==========================================
print("Đang tính Pearson correlation với RainTomorrow...")

# Encode target (RainTomorrow -> 0/1)
df_corr = df.withColumn(
    "RainTomorrow_num",
    when(col("RainTomorrow") == "Yes", 1).otherwise(0)
)

# Chọn numeric features để EDA
eda_features = [
    "MinTemp", "MaxTemp", "Rainfall", "TempRange",
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    "Pressure9am", "Pressure3pm", "PressureDiff",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "WindSpeedDiff"
]

assembler_eda = VectorAssembler(
    inputCols=eda_features + ["RainTomorrow_num"],
    outputCol="eda_features"
)

df_eda_vector = assembler_eda.transform(df_corr).select("eda_features")

corr_matrix = Correlation.corr(df_eda_vector, "eda_features", "pearson").head()[0]

# Convert sang pandas để đọc dễ hơn
corr_array = corr_matrix.toArray()
cols = eda_features + ["RainTomorrow_num"]

corr_df = pd.DataFrame(corr_array, columns=cols, index=cols)

print("\n===== CORRELATION WITH RainTomorrow =====")
print(corr_df["RainTomorrow_num"].sort_values(ascending=False))


# ==========================================
# PHẦN 3: VECTOR ASSEMBLER (ML PIPELINE)
# ==========================================
print("Đang gom nhóm 22 biến độc lập...")

feature_cols = [
    "MinTemp", "Temp3pm", "TempRange",
    "Rainfall",
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    "Pressure9am", "PressureDiff",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "WindSpeedDiff",
    "Month",
    "Location_index", "WindGustDir_index",
    "WindDir9am_index", "WindDir3pm_index", "RainToday_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df)


# ==========================================
# PHẦN 4: STANDARD SCALER
# ==========================================
print("Đang chuẩn hóa dữ liệu...")

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=False
)

scaler_model = scaler.fit(df_vector)
df_final = scaler_model.transform(df_vector)


# ==========================================
# PHẦN 5: LƯU HDFS
# ==========================================
print("Đang lưu dữ liệu...")

df_save = df_final.select("scaled_features", "label")

hdfs_output_path = "hdfs://localhost:9000/DACK/weather_ml_rain_index"

df_save.coalesce(1) \
    .write \
    .mode("overwrite") \
    .parquet(hdfs_output_path)

print("========= THÀNH CÔNG =========")
print(f"Saved to: {hdfs_output_path}")