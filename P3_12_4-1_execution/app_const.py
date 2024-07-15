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
# LOG_DETAILS = {"SILENT": 0, "NORMAL": 1, "VERBOS": 2}


PROFILES = {
    "2": {
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "# IMAGE_SIZE": 128,
        "IMAGE_SIZE": 224,
        "LOSS_FUNCTION": "categorical_crossentropy",
        "LEARNING_RATE": 0.001,
        "FREEZE_MODEL": True,
        "DROP_OUT_RATE": 0.5,
    },
    "2.A": {
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "# IMAGE_SIZE": 128,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": {"SILENT": 0, "NORMAL": 1, "VERBOS": 2},
        "LOSS_FUNCTION": "categorical_crossentropy",
        "LEARNING_RATE": 0.001,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
    "2.B": {
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "# IMAGE_SIZE": 128,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": {"SILENT": 0, "NORMAL": 1, "VERBOS": 2},
        "LOSS_FUNCTION": "categorical_crossentropy",
        "# LEARNING_RATE": 0.001,
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
    "2.C": {
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "# IMAGE_SIZE": 128,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": {"SILENT": 0, "NORMAL": 1, "VERBOS": 2},
        "LOSS_FUNCTION": "categorical_crossentropy",
        "# LEARNING_RATE": 0.001,
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
    "2.D": {
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "# IMAGE_SIZE": 128,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": {"SILENT": 0, "NORMAL": 1, "VERBOS": 2},
        "LOSS_FUNCTION": "categorical_crossentropy",
        "# LEARNING_RATE": 0.001,
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
    "2.E": {
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "SEED": 42,
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "TRAIN_DECREASE_FACTOR": 10,
        "TRAIN_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "# IMAGE_SIZE": 128,
        "IMAGE_SIZE": 224,
        "LOG_DETAILS": {"SILENT": 0, "NORMAL": 1, "VERBOS": 2},
        "LOSS_FUNCTION": "categorical_crossentropy",
        "# LEARNING_RATE": 0.001,
        "LEARNING_RATE": 0.01,
        "FREEZE_MODEL": False,
        "DROP_OUT_RATE": 0.5,
    },
}
