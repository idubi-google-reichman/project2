from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import tensorflow as tf
import threading
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from keras.layers import Resizing
from tensorflow.keras import layers, models
import tensorflow as tf


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
socketio = SocketIO(app, cors_allowed_origins="*")

stop_training_flag = False
model = None


stop_training_flag = False
model = None


# Custom callback to handle stopping and logging
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global stop_training_flag
        if stop_training_flag:
            self.model.stop_training = True
        socketio.emit("log", {"epoch": epoch, "logs": logs})


@app.route("/train", methods=["POST"])
def train(image_size=128):
    global stop_training_flag, model
    stop_training_flag = False

    data = request.json
    epochs = data.get("epochs", 10)

    # Dummy data for demonstration
    (X_train, y_train), (X_val, y_val) = tf.keras.datasets.cifar10.load_data()
    X_train, X_val = X_train / 255.0, X_val / 255.0

    base_model = MobileNet(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze the base model

    # Define the model
    model = models.Sequential(
        [
            Resizing(
                image_size,
                image_size,
                interpolation="nearest",
                input_shape=X_train.shape[1:],
            ),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation="softmax"),
        ]
    )

    optimizer = Adam()

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    threading.Thread(
        target=lambda: model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[CustomCallback()],
        )
    ).start()

    return jsonify({"status": "Training started"}), 200


@app.route("/stop", methods=["POST"])
def stop():
    global stop_training_flag
    stop_training_flag = True
    return jsonify({"status": "Training stopping"}), 200


if __name__ == "__main__":
    socketio.run(app, debug=True)
