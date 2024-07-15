# import tensorflow as tf

# # tf.config.list_physical_devices("GPU")


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)


# Define the model
def create_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    return model


# Main function to run the model on GPU
def main():
    with tf.device("/GPU:0"):
        model = create_model()
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Train the model
        history = model.fit(
            x_train, y_train, epochs=10, validation_data=(x_test, y_test)
        )

        # Evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")


if __name__ == "__main__":
    main()
