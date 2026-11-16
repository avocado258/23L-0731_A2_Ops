import os
MODEL_DIR = os.getenv("MODEL_DIR", "models/champion")
THRESHOLD = os.getenv("THRESHOLD")                      # optional override of meta.json
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///history.db")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))