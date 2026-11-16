from dotenv import load_dotenv
from tensorflow import keras
import os, mlflow

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("toy_experiment")


(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
x_train, x_val = x_train/255.0, x_val/255.0

for lr in [1e-2, 1e-3]:
    with mlflow.start_run(run_name=f"lr_{lr}"):
        mlflow.log_param("lr", lr)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)
        for epoch, (loss, acc) in enumerate(zip(history.history["loss"], history.history["accuracy"])):
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
        import matplotlib.pyplot as plt
        plt.plot(history.history["loss"]); plt.savefig("toy_loss.png")
        mlflow.log_artifact("toy_loss.png")
        mlflow.keras.log_model(model, "model")