import os

MODEL_DIR = os.getenv("MODEL_DIR", "models/champion")
THRESHOLD = os.getenv("THRESHOLD")                      # optional override of meta.json
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///history.db")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "").rstrip("/")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
INFERENCE_API_TIMEOUT = int(os.getenv("INFERENCE_API_TIMEOUT", "120"))   # covers a cold start