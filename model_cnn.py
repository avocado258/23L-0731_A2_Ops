from dotenv import load_dotenv
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow, mlflow.keras
import subprocess, numpy as np, random
from pipeline import train_ds, val_ds

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

def build_cnn(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model

class MLflowLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            mlflow.log_metric(k, v, step=epoch)

if __name__ == "__main__":
    mlflow.set_experiment("visionguard")

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    with mlflow.start_run(run_name="baseline_cnn"):
        mlflow.log_params({
            "model": "baseline_cnn",
            "lr": 1e-3,
            "batch_size": 32,
            "img_size": 224,
            "seed": 42
        })
        mlflow.log_param("git_commit", git_commit)

        model = build_cnn()
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("best_cnn.keras", monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            MLflowLogger()
        ]

        history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
        best_epoch = np.argmin(history.history["val_loss"]) + 1
        mlflow.log_metric("best_epoch", best_epoch)

        mlflow.keras.log_model(model, "model")