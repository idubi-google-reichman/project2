import utils.logging_utils as logging_utils

# ---------------------------------------------------------------------------------------------------
#    Tensorflow documentation : https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
# 	"auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
#    auto" becomes 1 for most cases. Note that the progress bar is not particularly
#    useful when logged to a file, so verbose=2 is recommended when not running interactively
#   (e.g. in a production environment). Defaults to "auto".
# ---------------------------------------------------------------------------------------------------
VERBOSE = {"SILENT": 0, "PROGRESS_BAR": 1, "SINGLE_LINE": 2, "AUTO": "auto"}
LOSS_FUNCTION = "categorical_crossentropy"
DATA_IMAGES_POSITION = 0
DATA_LABELS_POSITION = 1

SEED = 42

LOG_LEVEL = "INFO"
EXECUTION_SET = "3,3.A,3.B,3.C,3.D,3.E"  # this is a string that will parse into set
EXECUTION_PATH = "./resources/executions/"
