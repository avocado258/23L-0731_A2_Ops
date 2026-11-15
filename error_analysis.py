# error_analysis.py
import os
from dotenv import load_dotenv
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

IMG_SIZE = 224

def load_and_predict(model_path, test_csv="data/test.csv"):
    model = tf.keras.models.load_model(model_path)
    df = pd.read_csv(test_csv)
    class_names = sorted(df["class"].unique())
    label_map = {c: i for i, c in enumerate(class_names)}
    df["label"] = df["class"].map(label_map)

    paths, y_true, y_prob = [], [], []
    for _, row in df.iterrows():
        img = tf.io.read_file(row["path"])
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.expand_dims(img, 0)
        prob = model.predict(img, verbose=0)[0][0]

        paths.append(row["path"])
        y_true.append(row["label"])
        y_prob.append(prob)

    return np.array(paths), np.array(y_true), np.array(y_prob), class_names

def error_analysis(model_path, run_name, threshold=0.5):
    paths, y_true, y_prob, class_names = load_and_predict(model_path)
    y_pred = (y_prob >= threshold).astype(int)

    misclassified_idx = np.where(y_pred != y_true)[0][:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, idx in zip(axes.flat, misclassified_idx):
        img = Image.open(paths[idx])
        ax.imshow(img)
        ax.axis("off")
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)

    plt.tight_layout()
    fig_path = f"figures/misclassified_{run_name}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(fig_path)
    plt.close()

    with mlflow.start_run(run_name=f"{run_name}_error_analysis"):
        mlflow.log_artifact(fig_path)

    print(f"Saved {len(misclassified_idx)} misclassified examples to {fig_path}")
    for idx in misclassified_idx:
        print(paths[idx], "true:", class_names[y_true[idx]], "pred:", class_names[y_pred[idx]])

    return paths[misclassified_idx], y_true[misclassified_idx], y_pred[misclassified_idx]

if __name__ == "__main__":
    error_analysis("best_cnn.keras", "baseline_cnn", threshold=0.57)
    error_analysis("best_transfer_stage2.keras", "transfer_finetune", threshold=0.42)