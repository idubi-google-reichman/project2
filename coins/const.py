from utils.logging_utils import LOG_LEVEL

COMET_CONFIG = ".comet.config"

LOG_LEVEL = LOG_LEVEL["INFO"]

coins_values = [
    {"01-Shekel-Coin": 1},
    {"10-Shekel-Coin": 10},
    {"02-Shekel-Coin": 2},
    {"05-Shekel-Coin": 5},
]

labels_values = {"0": 1, "1": 10, "2": 2, "3": 5}


EXECUTION_MODES = {
    "PREPARE_DATASET": 1,
    "TRAIN": 2,
    "VALIDATE": 3,
    "EVALUATE": 4,
    "PREDICT": 5,
}
RELATIVE_DATASET_BASE_PATH = "./resources/base_dataset"
RELATIVE_DATASET_PATH = "./resources/datasets"
ALL_EXECUTION_MODES = ["prepare-dataset", "train", "validate", "evaluate", "predict"]
DATASET_PATH = "./resources/datasets/data.yaml"
# DATASET_PATH = "data.yaml"
EXECUTION_PATH = "./resources/executions/"


VALIDATION_PCT = 15
TRAIN_PCT = 70
DATASET_USE_PCT = 100

BATCH = 16
EPOCHS = 40
LEARNING_RATE = 0.000000001

PROJECT_PATH = "."
