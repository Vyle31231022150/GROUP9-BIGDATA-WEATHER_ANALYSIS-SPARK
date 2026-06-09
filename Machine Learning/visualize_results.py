import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("encoding_comparison_results.csv")

labels = [
    f"{row['Model']}\n({row['Encoding'].replace(' Encoding','')})"
    for _, row in df.iterrows()
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12,6))

plt.bar(
    x - width/2,
    df["ROC_AUC"],
    width,
    label="ROC-AUC"
)

plt.bar(
    x + width/2,
    df["F1_Score"],
    width,
    label="F1-Score"
)

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Performance Comparison of Machine Learning Models")
plt.legend()

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()