import argparse
from dotenv import load_dotenv
import os, subprocess, numpy as np, random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import mlflow, mlflow.keras
import pandas as pd, yaml

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

AUTOTUNE = tf.data.AUTOTUNE

def build_pipeline(img_size, batch_size, use_augmentation):
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    def make_ds(csv_path, training=False):
        df = pd.read_csv(csv_path)
        class_names = sorted(df["class"].unique())
        label_map = {c: i for i, c in enumerate(class_names)}
        df["label"] = df["class"].map(label_map)
        ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.shuffle(1000, seed=42)
            if use_augmentation:
                ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds("data/train.csv", training=True)
    val_ds = make_ds("data/val.csv")
    return train_ds, val_ds

def build_model(backbone, lr, n_unfrozen, img_size):
    if backbone == "mobilenet":
        base = MobileNetV2(input_shape=(img_size,img_size,3), include_top=False, weights="imagenet")
        preprocess = mobilenet_preprocess
    elif backbone == "efficientnet":
        base = EfficientNetB0(input_shape=(img_size,img_size,3), include_top=False, weights="imagenet")
        preprocess = efficientnet_preprocess
    else:
        raise ValueError("unknown backbone")

    base.trainable = True
    if n_unfrozen == 0:
        base.trainable = False
    else:
        for layer in base.layers[:-n_unfrozen]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(img_size,img_size,3))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model

def run_experiment(run_name, backbone="mobilenet", lr=1e-4, n_unfrozen=20,
                    img_size=224, batch_size=32, use_augmentation=True, epochs=10):
    tf.random.set_seed(42); np.random.seed(42); random.seed(42)

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    with open("data/raw.dvc") as f:
        dvc_hash = yaml.safe_load(f)["outs"][0]["md5"]

    train_ds, val_ds = build_pipeline(img_size, batch_size, use_augmentation)
    model = build_model(backbone, lr, n_unfrozen, img_size)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "backbone": backbone, "lr": lr, "unfrozen_layers": n_unfrozen,
            "img_size": img_size, "batch_size": batch_size,
            "augmentation": use_augmentation
        })
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("seed", 42)
        mlflow.set_tag("dvc_data_version", dvc_hash)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

        for k, v in history.history.items():
            for epoch, val in enumerate(v):
                mlflow.log_metric(k, val, step=epoch)

        mlflow.keras.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--backbone", default="mobilenet")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_unfrozen", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augmentation", type=lambda x: x.lower()=="true", default=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_id = run_experiment(args.run_name, args.backbone, args.lr, args.n_unfrozen,
                             args.img_size, args.batch_size, args.augmentation, args.epochs)
    print(f"Run ID: {run_id}")