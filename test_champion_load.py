from dotenv import load_dotenv
import os
import mlflow
import numpy as np
from PIL import Image

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

model = mlflow.keras.load_model("models:/visionguard-model@champion")

img_path = "data/raw/AiArtData/AiArtData/2K65J19-2.jpg"
img = Image.open(img_path).convert("RGB").resize((224, 224))
arr = np.expand_dims(np.array(img), axis=0).astype("float32")

pred = model.predict(arr)
print("Prediction (probability of AI-generated):", pred)
