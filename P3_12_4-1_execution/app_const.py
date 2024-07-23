from logger import logging

LOG_LEVEL = logging.INFO

DATASET_NAME = "cifar10"
SEED = 42
TRAIN_VALIDATION_SPLIT = 0.2
TRAIN_DECREASE_FACTOR = 10
TRAIN_EPOCHS = 5
BATCH_SIZE = 32
# IMAGE_SIZE = 128
IMAGE_SIZE = 224
# ---------------------------------------------------------------------------------------------------
#    Tensorflow documentation : https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
# 	"auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
#    auto" becomes 1 for most cases. Note that the progress bar is not particularly
#    useful when logged to a file, so verbose=2 is recommended when not running interactively
#   (e.g. in a production environment). Defaults to "auto".
# ---------------------------------------------------------------------------------------------------
VERBOSE = {"SILENT": 0, "PROGRESS_BAR": 1, "SINGLE_LINE": 2, "AUTO": "auto"}

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
