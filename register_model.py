import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

best_model_id = "m-8ac9228f2750423f9f14a53e62dcc33f"  # cmp_efficientnet
second_model_id = "m-41aa5fa0e86a4cd895b79f04eb603eb1"  # cmp_smaller_img

result1 = mlflow.register_model(f"models:/{best_model_id}", "visionguard-model")
print(f"Registered version {result1.version} from model {best_model_id}")

result2 = mlflow.register_model(f"models:/{second_model_id}", "visionguard-model")
print(f"Registered version {result2.version} from model {second_model_id}")

client = MlflowClient()
client.set_registered_model_alias("visionguard-model", "champion", result1.version)
print(f"Alias 'champion' set to version {result1.version}")