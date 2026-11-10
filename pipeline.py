import tensorflow as tf
import pandas as pd

IMG_SIZE = 224  # justify in report: matches MobileNetV2/common pretrained input size
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])  # distorts aspect ratio — state this explicitly
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

def make_dataset(csv_path, training=False):
    df = pd.read_csv(csv_path)
    class_names = sorted(df["class"].unique())
    label_map = {c: i for i, c in enumerate(class_names)}
    df["label"] = df["class"].map(label_map)

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(1000, seed=42)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, class_names

train_ds, class_names = make_dataset("data/train.csv", training=True)
val_ds, _ = make_dataset("data/val.csv")
test_ds, _ = make_dataset("data/test.csv")