import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pipeline import train_ds, val_ds

def build_vg_model(hp):
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    n_unfrozen = hp.Int("unfrozen_layers", 0, 40, step=10)
    base.trainable = True
    for layer in base.layers[:-n_unfrozen] if n_unfrozen > 0 else base.layers:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(224,224,3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1))(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("lr", [1e-3, 1e-4, 1e-5])),
        loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    return model

if __name__ == "__main__":
    tuner = kt.Hyperband(build_vg_model, objective="val_accuracy", max_epochs=10, directory="tuner_vg")
    tuner.search(train_ds, validation_data=val_ds, epochs=10)

    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:", best_hp.values)

    print("\nTrial summary:")
    tuner.results_summary()