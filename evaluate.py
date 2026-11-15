import os
from dotenv import load_dotenv
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from pipeline import val_ds, test_ds

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

os.makedirs("figures", exist_ok=True)

def evaluate_model(model_path, run_name):
    model = tf.keras.models.load_model(model_path)

    # --- collect validation predictions (for threshold selection) ---
    val_true, val_prob = [], []
    for imgs, labels in val_ds:
        val_prob.extend(model.predict(imgs, verbose=0).flatten())
        val_true.extend(labels.numpy())
    val_true, val_prob = np.array(val_true), np.array(val_prob)

    fpr, tpr, thresholds = roc_curve(val_true, val_prob)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    # --- collect test predictions ---
    y_true, y_prob = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0).flatten()
        y_prob.extend(preds)
        y_true.extend(labels.numpy())
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred),
        "test_recall": recall_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_prob),
    }

    print(f"\n=== {run_name} ===")
    print("Threshold:", best_thresh)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    with mlflow.start_run(run_name=f"{run_name}_eval"):
        mlflow.log_param("decision_threshold", float(best_thresh))
        mlflow.log_metrics(metrics)

        ConfusionMatrixDisplay(cm).plot()
        cm_path = f"figures/confusion_matrix_{run_name}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        plt.figure()
        test_fpr, test_tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(test_fpr, test_tpr)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {run_name}")
        roc_path = f"figures/roc_curve_{run_name}.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)

        mlflow.log_artifact("data/train.csv")
        mlflow.log_artifact("data/val.csv")
        mlflow.log_artifact("data/test.csv")
        mlflow.log_artifact("requirements.txt")

    return y_true, y_pred, y_prob, best_thresh

if __name__ == "__main__":
    print("Evaluating baseline CNN...")
    evaluate_model("best_cnn.keras", "baseline_cnn")

    print("\nEvaluating transfer model (stage 2)...")
    evaluate_model("best_transfer_stage2.keras", "transfer_finetune")