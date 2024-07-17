DATASET_NAME = "cifar10"
SEED = 42
TRAIN_VALIDATION_SPLIT = 0.2
TRAIN_DECREASE_FACTOR = 10
TRAIN_EPOCHS = 5
BATCH_SIZE = 32
# IMAGE_SIZE = 128
IMAGE_SIZE = 224
LOG_DETAILS = {"SILENT": 0, "NORMAL": 1, "VERBOS": 2}
LOSS_FUNCTION = "categorical_crossentropy"
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.01
FREEZE_MODEL = False
DATA_IMAGES_POSITION = 0
DATA_LABELS_POSITION = 1
DROP_OUT_RATE = 0.5


DATA_IMAGES_POSITION = 0
DATA_LABELS_POSITION = 1
SEED = 42


PROFILES = {
    "2": {
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
        "FREEZE_MODEL": True,
        "LOG_DETAILS": "VERBOS",
        "DROP_OUT_RATE": 0.5,
    },
    "2.A": {
        "NAME": "2.A",
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "IMAGE_SIZE": 224,
        "LOSS_FUNCTION": "categorical_crossentropy",
        "LEARNING_RATE": 0.001,
        "FREEZE_MODEL": False,
        "LOG_DETAILS": "VERBOS",
        "DROP_OUT_RATE": 0.5,
    },
    "2.B": {
        "NAME": "2.B",
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": "VERBOS",
        "LOSS_FUNCTION": "categorical_crossentropy",
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
    "2.C": {
        "NAME": "2.C",
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": "VERBOS",
        "LOSS_FUNCTION": "categorical_crossentropy",
        "# LEARNING_RATE": 0.001,
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
    "2.D": {
        "NAME": "2.D",
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": "VERBOS",
        "LOSS_FUNCTION": "categorical_crossentropy",
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
    "2.E": {
        "NAME": "2.E",
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": "VERBOS",
        "LOSS_FUNCTION": "categorical_crossentropy",
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
}
