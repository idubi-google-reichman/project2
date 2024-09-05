from utils.logging_utils import LOG_LEVEL

COMET_CONFIG = ".comet.config"

LOG_LEVEL = LOG_LEVEL["INFO"]
EXECUTION_MODES = {
    "PREPARE_DATASET": 1,
    "TRAIN": 2,
    "VALIDATE": 3,
    "EVALUATE": 4,
    "PREDICT": 5,
}

ALL_EXECUTION_MODES = "prepare-dataset,train,validate,evaluate,predict"
DATASET_PATH = "./resources/datasets/data.yaml"
# DATASET_PATH = "data.yaml"
EXECUTION_PATH = "./resources/executions/"

VALIDATION_PCT = 15
TRAIN_PCT = 70
DATASET_USE_PCT = 100

BATCH = 16
EPOCHS = 40
LEARNING_RATE = 0.00001


PROJECT_PATH = "."
