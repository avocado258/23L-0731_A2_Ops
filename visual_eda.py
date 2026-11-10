import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/audit.csv")
df = df[~df["corrupt"]]

# sample grid per class
for cls in df["class"].unique():
    sample = df[df["class"]==cls].sample(9, random_state=42)
    fig, axes = plt.subplots(3,3, figsize=(8,8))
    for ax, (_, row) in zip(axes.flat, sample.iterrows()):
        ax.imshow(Image.open(row["path"]))
        ax.axis("off")
    fig.suptitle(f"Sample: {cls}")
    plt.savefig(f"figures/sample_{cls}.png")

# histograms
for col in ["width", "height", "file_size"]:
    plt.figure()
    for cls in df["class"].unique():
        plt.hist(df[df["class"]==cls][col], bins=40, alpha=0.5, label=cls)
    plt.legend(); plt.title(f"{col} distribution")
    plt.savefig(f"figures/hist_{col}.png")