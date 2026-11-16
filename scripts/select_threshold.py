import json, pathlib, sys
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.insert(0, ".")
from app.inference import preprocess_bytes

D = pathlib.Path("models/champion")
model = tf.keras.models.load_model(D / "model.keras")
df = pd.read_csv("data/val.csv")

label_map = {c: i for i, c in enumerate(sorted(df["class"].unique()))}
assert label_map == {"AiArtData": 0, "RealArt": 1}, label_map

y = df["class"].map(label_map).values
p = np.array([model.predict(preprocess_bytes(open(f, "rb").read())[None], verbose=0)[0, 0]
              for f in df["path"]])

auc = roc_auc_score(y, p)
assert auc > 0.5, f"AUC={auc:.3f}: labels/preprocessing likely flipped"
print("mean p(real) | RealArt:", p[y == 1].mean(), "| AiArtData:", p[y == 0].mean())

fpr, tpr, thr = roc_curve(y, p)
t = float(thr[np.argmax(tpr - fpr)])          # Youden's J
print(f"val AUC={auc:.4f}  threshold={t:.4f}  J={np.max(tpr - fpr):.4f}")

meta = json.loads((D / "meta.json").read_text())
meta.update(threshold=t, label_map=label_map, threshold_method="youden_j_val", val_auc=auc)
(D / "meta.json").write_text(json.dumps(meta, indent=2))