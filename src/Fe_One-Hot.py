from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    OneHotEncoder,
    StringIndexer
)
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# PHẦN 1: KHỞI TẠO VÀ ĐỌC DỮ LIỆU
# ==========================================
spark = SparkSession.builder \
    .appName("Weather_Feature_Engineering_OHE") \
    .master("local[*]") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

print("Đang đọc dữ liệu sạch...")

df = spark.read.parquet("hdfs://master:9000/DACK/weather_clean")

print(f"Số dòng dữ liệu: {df.count()}")

# ==========================================
# PHẦN 1.5: TẠO LABEL (FIX QUAN TRỌNG)
# ==========================================
label_indexer = StringIndexer(
    inputCol="RainTomorrow",
    outputCol="label"
)

df = label_indexer.fit(df).transform(df)

# ==========================================
# PHẦN 2: FEATURE ENGINEERING
# ==========================================

df = df.withColumn("TempRange", col("MaxTemp") - col("MinTemp"))
df = df.withColumn("HumidityDiff", col("Humidity9am") - col("Humidity3pm"))
df = df.withColumn("PressureDiff", col("Pressure9am") - col("Pressure3pm"))
df = df.withColumn("WindSpeedDiff", col("WindSpeed3pm") - col("WindSpeed9am"))
df.cache()
# ==========================================
# PHẦN 3.5A: CORRELATION
# ==========================================

numeric_cols = [
    "MinTemp", "MaxTemp", "Temp9am", "Temp3pm",
    "TempRange", "Rainfall",
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    "Pressure9am", "Pressure3pm", "PressureDiff",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "WindSpeedDiff", "Month"
]

pdf = df.select(numeric_cols).toPandas()

corr_matrix = pdf.corr()

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

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

plt.title("Feature Correlation Matrix")
plt.show()

# ==========================================
# PHẦN 3.6: MUTUAL INFORMATION
# ==========================================

high_corr_features = [
    "MinTemp", "MaxTemp", "Temp9am",
    "Temp3pm", "Pressure9am", "Pressure3pm"
]

pdf = df.select(high_corr_features + ["label"]).toPandas()

X = pdf.drop(columns=["label"])
y = pdf["label"]

mi_scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI Score": mi_scores
}).sort_values(by="MI Score", ascending=False)

print(mi_df)

plt.figure(figsize=(8,5))
sns.barplot(data=mi_df, x="MI Score", y="Feature")
plt.title("Mutual Information")
plt.show()

# ==========================================
# PHẦN 4: STRING INDEXER (FIX QUAN TRỌNG)
# ==========================================

categorical_cols = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday"
]

for c in categorical_cols:
    indexer = StringIndexer(
        inputCol=c,
        outputCol=c + "_index"
    )
    df = indexer.fit(df).transform(df)

# ==========================================
# PHẦN 5: ONE HOT ENCODING
# ==========================================

index_cols = [
    "Location_index",
    "WindGustDir_index",
    "WindDir9am_index",
    "WindDir3pm_index",
    "RainToday_index"
]

ohe_cols = [
    "Location_ohe",
    "WindGustDir_ohe",
    "WindDir9am_ohe",
    "WindDir3pm_ohe",
    "RainToday_ohe"
]

encoder = OneHotEncoder(
    inputCols=index_cols,
    outputCols=ohe_cols,
    dropLast=True
)

df = encoder.fit(df).transform(df)

# ==========================================
# PHẦN 6: VECTOR ASSEMBLER
# ==========================================

feature_cols = [
    "MinTemp", "Temp3pm", "TempRange",
    "Rainfall",
    "Humidity9am", "Humidity3pm", "HumidityDiff",
    "Pressure9am", "PressureDiff",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "WindSpeedDiff",
    "Month",
    "Location_ohe",
    "WindGustDir_ohe",
    "WindDir9am_ohe",
    "WindDir3pm_ohe",
    "RainToday_ohe"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_vector = assembler.transform(df)

# ==========================================
# PHẦN 7: STANDARD SCALER
# ==========================================

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=False
)

df_final = scaler.fit(df_vector).transform(df_vector)

# ==========================================
# PHẦN 8: SAVE DATASET
# ==========================================

df_save = df_final.select("scaled_features", "label")

output_path = "hdfs://master:9000/DACK/weather_ml_rain_ohe"

df_save.coalesce(1).write.mode("overwrite").parquet(output_path)

print("DONE SAVE:", output_path)
df.unpersist()

spark.stop()