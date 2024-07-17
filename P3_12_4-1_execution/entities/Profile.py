import json
from abc import ABC, abstractmethod


class Profile(ABC):
    def __init__(self, profile_data):
        # Parse JSON string if provided as string
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)

        self.name = profile_data.get("NAME")
        self.model_name = profile_data.get("MODEL_NAME")
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
        self.log_details = profile_data.get("LOG_DETAILS")
        self.drop_out_rate = profile_data.get("DROP_OUT_RATE")
        self.optimizer = profile_data.get("OPTIMIZER")
        self.model_structure = profile_data.get("MODEL_STRUCTURE")

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def set_optimizer(self):
        pass

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
            "LOG_DETAILS": self.log_details,
            "DROP_OUT_RATE": self.drop_out_rate,
            "OPTIMIZER": self.optimizer,
            "MODEL_STRUCTURE": self.model_structure,
        }
        return summary

    # Setters and Getters
    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_model_name(self, model_name):
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name

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

    def set_log_details(self, log_details):
        self.log_details = log_details

    def get_log_details(self):
        return self.log_details

    def set_drop_out_rate(self, drop_out_rate):
        self.drop_out_rate = drop_out_rate

    def get_drop_out_rate(self):
        return self.drop_out_rate

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_optimizer(self):
        return self.optimizer

    def set_model_structure(self, model_structure):
        self.model_structure = model_structure

    def get_model_structure(self):
        return self.model_structure


# Example of a child class that implements the abstract methods
class MobileNetProfile(Profile):
    def train(self):
        print("Training MobileNet model...")

    def create_model(self):
        print("Creating MobileNet model...")

    def set_optimizer(self):
        print("Setting optimizer for MobileNet model...")


# Usage example
profile_json = """
{
    "NAME": "2",
    "MODEL_NAME": "MobileNet",
    "DATASET_NAME": "cifar10",
    "TRAIN_VALIDATION_SPLIT": 0.2,
    "SEED": 42,
    "TRAIN_DECREASE_FACTOR": 80,
    "TRAIN_EPOCHS": 1,
    "BATCH_SIZE": 32,
    "IMAGE_SIZE": 224,
    "LOSS_FUNCTION": "categorical_crossentropy",
    "LEARNING_RATE": 0.001,
    "FREEZE_MODEL": true,
    "LOG_DETAILS": "VERBOS",
    "DROP_OUT_RATE": 0.5,
    "OPTIMIZER": "adam",
    "MODEL_STRUCTURE": "complex"
}
"""

profile = MobileNetProfile(profile_json)
print(profile.show_summary())

# Example method calls
profile.train()
profile.create_model()
profile.set_optimizer()
