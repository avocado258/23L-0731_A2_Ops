import json, os, pathlib
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
NAME, ALIAS = "visionguard-model", "champion"
OUT = pathlib.Path(os.getenv("MODEL_DIR", "models/champion"))
OUT.mkdir(parents=True, exist_ok=True)

uri = os.environ.get("MLFLOW_TRACKING_URI", "")
if not uri:
    raise SystemExit("MLFLOW_TRACKING_URI is empty - check the GitHub Actions secret")
mlflow.set_tracking_uri(uri)
mv = MlflowClient().get_model_version_by_alias(NAME, ALIAS)
model = mlflow.keras.load_model(f"models:/{NAME}@{ALIAS}")
model.save(OUT / "model.keras")

meta_path = OUT / "meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
meta.update(model_name=NAME, model_version=str(mv.version), run_id=mv.run_id,
            alias=ALIAS, input_size=224)
meta_path.write_text(json.dumps(meta, indent=2))
print(meta)