import keras_tuner as kt
from tensorflow import keras

(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
x_train, x_val = x_train/255.0, x_val/255.0

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=5, directory="tuner_toy")
tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=5)
print(tuner.get_best_hyperparameters()[0].values)
tuner.results_summary()