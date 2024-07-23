import json
from abc import ABC, abstractmethod
from enum import Enum, auto
import app_const as const
import os
import random
import numpy as np
import app_const as const
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from keras.layers import Resizing, Dense, Dropout, RandomFlip, GlobalAveragePooling2D
from entities.ENeuralNetworkError import (
    ENeuralNetworkError,
    ENeuralNetworkProcessError,
    ENeuralNetworkArchitectureError,
)
from utils.timing_utils import capture_and_time, convert_count_time_to_human
from utils.model_utils import get_optimizer
from utils.logging_utils import LoggerUtility, LOG_LEVEL
from utils.plot_utils import plot_training_history
from prettytable import PrettyTable


class ModelStatus(Enum):
    INITIATED = auto()
    BUILT = auto()
    COMPILED = auto()
    TRAINED = auto()
    EVALUATED = auto()


class Profile(ABC):
    def __init__(self, profile_data):
        # Parse JSON string if provided as string
        self.model = None
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)

        self.init_all_fields_from_json(profile_data=profile_data)

        self.status = ModelStatus.INITIATED

        self.__build_model()

    # Set the random seeds - if no seed is provided,
    # the default value from app_const.py is used
    def set_seed(seed=const.SEED):
        # Set the random seeds
        os.environ["PYTHONHASHSEED"] = str(
            seed
        )  # This variable influences the hash function's behavior in Python 3.3 and later.
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        return

    def compile_model(self):
        if self.status.value < ModelStatus.BUILT.value:
            raise ENeuralNetworkProcessError(
                "Model is not built. Please build the model before compiling."
            )
        optimizer = get_optimizer(self.optimizer_name, self.learning_rate)
        loss_function = self.loss_function
        device = self.device_name
        with tf.device(device):
            model_compile_time = capture_and_time(
                func=self.model.compile,
                optimizer=optimizer,
                loss=loss_function,
                metrics=["accuracy"],
            )
        self.set_status(ModelStatus.COMPILED)
        return model_compile_time, convert_count_time_to_human(model_compile_time)

    def train_model(self, train_data, validation_data):
        if self.status.value < ModelStatus.COMPILED.value:
            raise ENeuralNetworkProcessError(
                "Model is not compiled. Please compile the model before training."
            )
        device = self.device_name
        epochs = self.train_epochs
        batch_size = self.batch_size
        if self.history is not None:
            del self.history

        with tf.device(device):
            history, model_train_time = capture_and_time(
                func=self.model.fit,
                x=train_data[const.DATA_IMAGES_POSITION],
                y=train_data[const.DATA_LABELS_POSITION],
                epochs=epochs,
                validation_data=validation_data,
                batch_size=batch_size,
            )

            history.history["model_train_time"] = model_train_time

            self.history = history
            self.set_status(ModelStatus.TRAINED)

        return (
            history,
            convert_count_time_to_human(model_train_time),
        )

    def evaluate_model(self, test_data, verbose=const.VERBOSE["SINGLE_LINE"]):
        if self.status.value < ModelStatus.TRAINED.value:
            raise ENeuralNetworkProcessError(
                "Model is not trained. Please train the model before evaluating."
            )
        device = self.device_name
        with tf.device(device):
            (test_loss, test_accuracy), evaluation_time = capture_and_time(
                func=self.model.evaluate,
                x=test_data[const.DATA_IMAGES_POSITION],
                y=test_data[const.DATA_LABELS_POSITION],
                verbose=verbose,
            )
            self.history.history["test_accuracy"] = test_accuracy
            self.history.history["test_loss"] = test_loss

            self.set_status(ModelStatus.EVALUATED)

        return (
            test_loss,
            test_accuracy,
            convert_count_time_to_human(evaluation_time),
        )

    def plot_training_history(self, save_path=None):
        if self.status.value < ModelStatus.TRAINED.value:
            raise ENeuralNetworkProcessError(
                "Model is not trained. Please train the model before plotting."
            )
        plot_training_history(self.history, save_path)

    def show_summary(self):

        summary = {
            "NAME": self.name,
            "MODEL_NAME": self.model_name,
            "DATASET_NAME": self.dataset_name,
            "TRAIN_VALIDATION_SPLIT": self.train_validation_split,
            "SEED": self.seed,
            "TRAIN_DECREASE_FACTOR": self.train_decrease_factor,
            "TRAIN_EPOCHS": self.train_epochs,
            "BATCH_SIZE": self.batch_size,
            "IMAGE_SIZE": self.image_size,
            "LOSS_FUNCTION": self.loss_function,
            "LEARNING_RATE": self.learning_rate,
            "FREEZE_MODEL": self.freeze_model,
            "VERBOSE_LOG_DETAILS": self.verbose,
            "OPTIMIZER": self.optimizer_name,
            "MODEL CONFIG": self.model_config,
        }

        #  this values are added to history after training and evaluating
        # if they are missiong this means that process is not complete
        # and need to train and execute evaluate before calling this method
        if self.status.value < ModelStatus.TRAINED.value:
            LoggerUtility.log_message(
                f"profile_{self.name}",
                f"Model is not trained. all history data and evaluations are not relevant and will not be seen ",
                LOG_LEVEL["WARNING"],
            )
        else:
            history_dict = self.history.history
            model_train_time = history_dict["model_train_time"]

            # Extract the last epoch's metrics from the history
            val_accuracy = history_dict["val_accuracy"][-1]
            val_loss = history_dict["val_loss"][-1]

            #  checking evaluated
            if self.status.value < ModelStatus.EVALUATED.value:
                LoggerUtility.log_message(
                    f"profile_{self.name}",
                    f"Model is not evaluated. all evaluation data is not relevant and will not be seen ",
                    LOG_LEVEL["WARNING"],
                )
            else:
                test_accuracy = history_dict["test_accuracy"]
                test_loss = history_dict["test_loss"]
            # until here evaluated

            table = PrettyTable()
            # Print the organized table
            table.field_names = [
                "Validation",
                "Test",
                "Inference",
            ]
            sub_headers = [
                "acc | loss",
                "acc | loss",
                "Time (sec)",
            ]
            table.add_row(sub_headers)
            table.add_row(["-" * 22, "-" * 22, "_" * 10])

            if self.status.value < ModelStatus.EVALUATED.value:
                evaluation_data = ("Not Trained" | "Not Trained",)
            else:
                evaluation_data = (f"{test_accuracy:.4f} | {test_loss:.4f}",)

            table.add_row(
                [
                    f"{val_accuracy:.4f} | {val_loss:.4f}",
                    f"{evaluation_data}",
                    f"{convert_count_time_to_human(model_train_time)}",
                ]
            )

        print(" ===================================")

        # until here trained is mandatory

        print(" ===================================")
        print(" execution summery for profile : ", self.name)
        print(" ===================================")

        print(table)

        print(" validation split : ", self.train_validation_split)
        print(" reduce factor    : ", self.train_decrease_factor)
        print(" number of epocs  : ", self.train_epochs)
        print(" batch size       : ", self.batch_size)
        print(" learning rate    : ", self.learning_rate)
        print(" is freeze model  : ", self.freeze_model)
        print(" loss function    : ", self.loss_function)
        print(" dataset name     : ", self.dataset_name)
        print(" random seed      : ", self.seed)

    # region __build_model

    def __bind_layer_params(self, layer_params):
        for key, value in layer_params.items():
            if isinstance(value, str) and value.startswith(":") and value.endswith(":"):
                attribute_name = value[1:-1]
                layer_params[key] = getattr(self, attribute_name)
        return

    def __build_model(self):

        if self.model:
            del self.model

        self.model = models.Sequential()

        base_model = self.get_base_model()

        self.model.trainable = not self.freeze_model

        self.generate_model(
            base_model=base_model, model_config=self.model_config["layers"]
        )

        self.set_status(ModelStatus.BUILT)

    # endregion __build_model

    # region generate_model

    #      region generate_model_description
    #      in subclass ths can be override or even called after performing something else
    #      this method take the configuration of the model and generate the model of it according to the configuration
    #      for instance :
    #      {MODEL_CONFIG: {
    #          input_shape: (32, 32, 3),
    #          layers: [
    #              {
    #                  type: "Resizing",
    #                  params: {
    #                      height: 224,
    #                      width: 224,
    #                      interpolation: "nearest",
    #                      input_shape:   ":resize_train_images_shape:",
    #                  },
    #              },
    #              {type: "BaseModel",              params: {}},
    #              {type: "RandomFlip",             params: {
    #                                                        mode: "horizontal_and_vertical"
    #                                                       }
    #              },
    #              {type: "GlobalAveragePooling2D", params: {}},
    #              {type: "Dropout",                params: {rate: 0.5}},
    #              {type: "Dense",                  params: {
    #                                                        units: 10,
    #                                                        activation: "softmax"
    #                                                       }
    #              },
    #          ],
    #      },
    #      endregion generate_model_description

    def generate_model(self, base_model, model_config):

        for layer_config in self.model_config["layers"]:
            layer_type = layer_config["type"]
            layer_params = layer_config["params"]
            self.__bind_layer_params(layer_params)

            if layer_type == "BaseModel":
                self.model.add(base_model)
            elif layer_type == "Resizing":
                self.model.add(layers.Resizing(**layer_params))
            elif layer_type == "RandomFlip":
                self.model.add(layers.RandomFlip(**layer_params))
            elif layer_type == "GlobalAveragePooling2D":
                self.model.add(layers.GlobalAveragePooling2D(**layer_params))
            elif layer_type == "Dropout":
                self.model.add(layers.Dropout(**layer_params))
            elif layer_type == "Dense":
                self.model.add(layers.Dense(**layer_params))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            LoggerUtility.log_message(
                f"profile_{self.name}",
                f"adding model layer : {layer_type}",
                LOG_LEVEL["DEBUG"],
            )

    # endregion generate_model

    # region  initiate the Profile class from profile data json object
    def init_all_fields_from_json(self, profile_data):
        self.name = profile_data.get("NAME")

        self.model_name = profile_data.get("MODEL_NAME")
        self.sub_model_name = profile_data.get("SUB_MODEL_NAME")
        self.dataset_name = profile_data.get("DATASET_NAME")
        self.train_validation_split = profile_data.get("TRAIN_VALIDATION_SPLIT")
        self.seed = profile_data.get("SEED")
        self.train_decrease_factor = profile_data.get("TRAIN_DECREASE_FACTOR")
        self.train_epochs = profile_data.get("TRAIN_EPOCHS")
        self.batch_size = profile_data.get("BATCH_SIZE")
        self.image_size = profile_data.get("IMAGE_SIZE")
        self.loss_function = profile_data.get("LOSS_FUNCTION")
        self.learning_rate = profile_data.get("LEARNING_RATE")
        self.freeze_model = profile_data.get("FREEZE_MODEL")
        self.verbose = const.VERBOSE[profile_data.get("VERBOSE_LOG_DETAILS")]
        # this value is set before creating the profile according to available devices
        self.device_name = profile_data.get("DEVICE_NAME")

        self.set_seed(self.seed)
        self.shape = profile_data.get("SHAPE") or (self.image_size, self.image_size, 3)
        # the shaoe in the resizing shape format
        self.resize_train_images_shape = profile_data.get("TRAIN_IMAGES_SHAPE")

        self.optimizer_name = profile_data.get("OPTIMIZER_NAME")
        self.model_config = profile_data.get("MODEL_CONFIG")
        self.history = []

    # endregion  init_all_fields_from_profile_data

    # region SETTERS AND GETTERS

    # Setters and Getters
    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_model_name(self, model_name):
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name

    def set_sub_model_name(self, sub_model_name):
        self.sub_model_name = sub_model_name

    def get_sub_model_name(self):
        return self.sub_model_name

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset_name(self):
        return self.dataset_name

    def set_train_validation_split(self, train_validation_split):
        self.train_validation_split = train_validation_split

    def get_train_validation_split(self):
        return self.train_validation_split

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed

    def set_train_decrease_factor(self, train_decrease_factor):
        self.train_decrease_factor = train_decrease_factor

    def get_train_decrease_factor(self):
        return self.train_decrease_factor

    def set_train_epochs(self, train_epochs):
        self.train_epochs = train_epochs

    def get_train_epochs(self):
        return self.train_epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def set_image_size(self, image_size):
        self.image_size = image_size

    def get_image_size(self):
        return self.image_size

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def get_loss_function(self):
        return self.loss_function

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.learning_rate

    def set_freeze_model(self, freeze_model):
        self.freeze_model = freeze_model

    def get_freeze_model(self):
        return self.freeze_model

    def set_verbose(self, verbose):
        self.verbose = verbose

    def get_verbose(self):
        return self.verbose

    def set_optimizer_name(self, optimizer, is_build_model=False):
        self.optimizer_name = optimizer
        is_build_model and self.build_model()

    def get_optimizer_name(self):
        return self.optimizer_name

    def set_model_config(self, model_config, is_rebuild_model=False):
        self.model_config = model_config
        is_rebuild_model and self.__build_model()

    def get_model_config(self):
        return self.model_config

    def set_shape(self, shape):
        self.shape = shape

    def get_shape(self):
        return self.shape

    def set_status(self, status):
        match status:
            case ModelStatus.INITIATED:
                self.status = status
            case ModelStatus.BUILT:
                del self.history
                self.history = []
                self.status = status
            case ModelStatus.COMPILED:
                if self.status.value == ModelStatus.INITIATED:
                    raise ENeuralNetworkProcessError(
                        f"current status ({self.get_status_name()}) not allow to change to COMPILED ({ModelStatus.COMPILED})"
                    )
                self.status = status
            case ModelStatus.TRAINED:
                if self.status.value < ModelStatus.COMPILED.value:
                    raise ENeuralNetworkProcessError(
                        f"current status ({self.get_status_name()}) not allow to change to TRAINED "
                    )
                self.status = status
            case ModelStatus.EVALUATED:
                if self.status.value < ModelStatus.TRAINED.value:
                    raise ENeuralNetworkProcessError(
                        f"current status ({self.get_status_name()}) not allow to change to EVALUATED "
                    )
                self.status = status
            case _:
                raise ENeuralNetworkError("Unknown model status")

    def get_status(self):
        return self.status

    # endregion SETTERS AND GETTERS
    def get_status_name(self):
        return self.status.name

    @abstractmethod
    def get_base_model(self):
        pass
