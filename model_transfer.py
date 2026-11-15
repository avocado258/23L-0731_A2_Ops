from dotenv import load_dotenv
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mlflow, mlflow.tensorflow
import subprocess, numpy as np, random
import matplotlib.pyplot as plt
from pipeline import train_ds, val_ds

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

def build_transfer_model():
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    base.trainable = False  # stage 1: frozen

    inputs = tf.keras.Input(shape=(224,224,3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    return model, base

def plot_history(history, stage_name, best_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].axvline(best_epoch - 1, color="red", linestyle="--", label="best epoch")
    axes[0].set_title(f"{stage_name} - Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].axvline(best_epoch - 1, color="red", linestyle="--", label="best epoch")
    axes[1].set_title(f"{stage_name} - Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = f"figures/transfer_{stage_name}.png"
    plt.savefig(fig_path)
    plt.close()
    return fig_path

if __name__ == "__main__":
    mlflow.tensorflow.autolog()

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    model, base = build_transfer_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy", "AUC"])

    print("Before unfreezing (stage 1):")
    print([l.name for l in base.layers if l.trainable])

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_transfer_stage1.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    with mlflow.start_run(run_name="transfer_stage1_frozen"):
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("seed", 42)
        history1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks_stage1)
        best_epoch1 = np.argmin(history1.history["val_loss"]) + 1
        mlflow.log_metric("best_epoch", best_epoch1)
        fig1_path = plot_history(history1, "stage1_frozen", best_epoch1)
        mlflow.log_artifact(fig1_path)

    # Stage 2: unfreeze last block, fine-tune with low LR
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    print("After unfreezing (stage 2):")
    print([l.name for l in base.layers if l.trainable])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy", metrics=["accuracy", "AUC"])

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_transfer_stage2.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    with mlflow.start_run(run_name="transfer_stage2_finetune"):
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("seed", 42)
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks_stage2)
        best_epoch2 = np.argmin(history2.history["val_loss"]) + 1
        mlflow.log_metric("best_epoch", best_epoch2)
        fig2_path = plot_history(history2, "stage2_finetune", best_epoch2)
        mlflow.log_artifact(fig2_path)