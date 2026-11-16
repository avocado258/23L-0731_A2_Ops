import io, json, pathlib, sys
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from app import config

SIZE = (224, 224)
IDX_TO_LABEL = {0: "AI-generated", 1: "Real"}   # AiArtData=0, RealArt=1 (asserted in select_threshold)


class ImageDecodeError(ValueError):
    pass


def preprocess_bytes(data: bytes) -> np.ndarray:
    """Same steps as training: RGB, 224x224 bilinear, EfficientNet preprocess_input."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()                                # forces full decode; catches truncation
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise ImageDecodeError(str(e)) from e
    arr = tf.image.resize(arr, SIZE, method="bilinear").numpy()
    return tf.keras.applications.efficientnet.preprocess_input(arr)


class Predictor:
    def __init__(self, model_dir=None):
        d = pathlib.Path(model_dir or config.MODEL_DIR)
        self.meta = json.loads((d / "meta.json").read_text())
        self.model = tf.keras.models.load_model(d / "model.keras")   # once, at startup
        self.threshold = float(config.THRESHOLD or self.meta["threshold"])
        self.model_name = f'{self.meta["model_name"]}:v{self.meta["model_version"]}'
        self.model.predict(np.zeros((1, *SIZE, 3), np.float32), verbose=0)   # warm-up

    def predict(self, data: bytes) -> dict:
        x = preprocess_bytes(data)[None]
        p_real = float(self.model.predict(x, verbose=0)[0, 0])
        idx = int(p_real >= self.threshold)
        return {
            "label": IDX_TO_LABEL[idx],
            "confidence": p_real if idx else 1.0 - p_real,
            "p_real": p_real,
            "threshold": self.threshold,
            "model": self.model_name,
        }


_predictor = None
def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


if __name__ == "__main__":      # console evidence: python -m app.inference a.jpg b.jpg
    pr = get_predictor()
    for f in sys.argv[1:]:
        print(f, pr.predict(open(f, "rb").read()))