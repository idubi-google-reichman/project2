from tensorflow.keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    RMSprop,
    Adadelta,
    Adamax,
    Nadam,
    Ftrl,
)
from tensorflow.keras.applications import MobileNet
from keras.layers import Resizing, Dense, Dropout, RandomFlip, GlobalAveragePooling2D
from tensorflow.keras import layers, models
import tensorflow as tf
import app_const as const


from classification_models.tfkeras import Classifiers


def set_model_seed(seed=const.SEED):
    tf.random.set_seed(seed)


def create_model(train_images, image_size=128, freeze_base_model=const.FREEZE_MODEL):
    # ResNet18, preprocess_input = Classifiers.get("resnet18")
    # base_model = ResNet18(
    #     input_shape=(224, 224, 3),
    #     weights="imagenet",
    #     include_top=False,
    #     classifier_activation="softmax",
    # )
    base_model = MobileNet(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = not (freeze_base_model)

    model = models.Sequential(
        [
            Resizing(
                image_size,
                image_size,
                interpolation="nearest",
                input_shape=train_images.shape[1:],
            ),
            base_model,
            RandomFlip(mode="horizontal_and_vertical"),
            GlobalAveragePooling2D(),
            Dropout(const.DROP_OUT_RATE),
            layers.Dense(10, activation="softmax"),
        ]
    )

    # Specify the learning rate

    # Instantiate the Adam optimizer with the default learning rate
    optimizer = Adam(learning_rate=const.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=const.LOSS_FUNCTION,
        metrics=["accuracy"],
    )

    return model


def train_model(model, train_data, validation_data, epochs, batch_size):
    history = model.fit(
        x=train_data[const.DATA_IMAGES_POSITION],
        y=train_data[const.DATA_LABELS_POSITION],
        epochs=epochs,
        validation_data=validation_data,
        batch_size=batch_size,
    )
    return history


def evaluate_model(model, test_data, verbose=const.VERBOSE["SINGLE_LINE"]):
    test_loss, test_acc = model.evaluate(
        test_data[const.DATA_IMAGES_POSITION],
        test_data[const.DATA_LABELS_POSITION],
        verbose=verbose,
    )
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
    return (test_loss, test_acc)


def get_optimizer(optimizer_name="adam", learning_rate=0.001):
    match optimizer_name:
        case "adam":
            return Adam(learning_rate=learning_rate)
        case "sgd" | "gradient_descent":
            return SGD(learning_rate=learning_rate)
        case "rmsprop":
            return RMSprop(learning_rate=learning_rate)
        case "adagrad":
            return Adagrad(learning_rate=learning_rate)
        case "adadelta":
            return Adadelta(learning_rate=learning_rate)
        case "adamax":
            return Adamax(learning_rate=learning_rate)
        case "nadam":
            return Nadam(learning_rate=learning_rate)
        case "ftrl":
            return Ftrl(learning_rate=learning_rate)
        case default:
            return Adam(learning_rate=learning_rate)  # default optimizer Adam
