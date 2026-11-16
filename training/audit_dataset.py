import os, hashlib
from PIL import Image
import pandas as pd

root = "data/raw"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
rows = []

for cls in os.listdir(root):
    cls_dir = os.path.join(root, cls)
    if not os.path.isdir(cls_dir):
        continue
    for dirpath, _, filenames in os.walk(cls_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in IMG_EXT:
                continue
            path = os.path.join(dirpath, fname)
            row = {"class": cls, "filename": fname, "path": path}
            try:
                size = os.path.getsize(path)
                with open(path, "rb") as f:
                    content = f.read()
                h = hashlib.sha256(content).hexdigest()
                with Image.open(path) as img:
                    img.verify()  # catches truncated files
                with Image.open(path) as img:
                    w, hgt = img.size
                    mode = img.mode
                    fmt = img.format
                row.update(file_size=size, width=w, height=hgt,
                           color_mode=mode, format=fmt, sha256=h, corrupt=False)
            except Exception as e:
                row.update(corrupt=True, error=str(e))
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/audit.csv", index=False)

print(df.groupby("class").size())
good = df[~df["corrupt"]]
print(good.groupby("class")[["width","height","file_size"]].agg(["min","median","max"]))
print("Corrupt:", df["corrupt"].sum())
dupes = good.duplicated(subset="sha256", keep=False)
print("Exact duplicates:", dupes.sum())