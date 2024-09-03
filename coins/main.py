import utils.comet_utils as COMET
from ultralytics import YOLO
import os

from datetime import datetime


import const
from utils.model_utils import is_cuda_enables
from utils.yolo_dataset_utils import prepare_dataset
from utils.logging_utils import LoggerUtility, LOG_LEVEL
from utils.plot_utils import plot_and_log_curves, plot_confusion_matrix

from app_args import init_args, get_parameter, get_weight_path

from commands.train import train
from utils.main_utils import get_execution_path


if __name__ == "__main__":
    args = init_args()
    if args.help_command:
        LoggerUtility.log_message("main - args", "")
        exit()

    command = get_parameter(args, "command")

    if (not command) or (command not in const.ALL_EXECUTION_MODES):
        print(f"none of the execution modes {const.ALL_EXECUTION_MODES} is recognized ")
        exit(-1)

    if command != "prepare-dataset":
        # create experiment that will be managed by COMIT
        experiment = COMET.create_experiment()

    execution_path = get_execution_path(args=args, command=command)

    LoggerUtility.configure_logger(
        log_level=LOG_LEVEL["INFO"],
        log_file_path=f"{execution_path}",
    )

    match command:
        case "prepare-dataset":
            _path_to_base = get_parameter(args, "path_to_base")
            _path_to_dataset = get_parameter(args, "path_to_dataset")
            _use_pct = int(get_parameter(args, "use_pct"))
            _train_pct = int(get_parameter(args, "train_pct"))
            _valid_pct = int(get_parameter(args, "valid_pct"))

            prepare_dataset(
                path_to_base=_path_to_base,
                path_to_dataset=_path_to_dataset,
                use_pct=_use_pct,
                train_pct=_train_pct,
                valid_pct=_valid_pct,
            )
        case "train":
            train(experiment=experiment, args=args)
        case "validate":
            is_cuda_enables()
            _data_path = get_parameter(args, "data_path")

            _weights = get_weight_path(args)

            if _weights and os.path.exists(_weights):
                model = YOLO(model=_weights)

            else:
                raise Exception(
                    "PARAMETER missing  or invalid : the weights path for VALIDATION is not valid"
                )
            if os.path.exists(_data_path):
                evaluation_results = model.val(data=_data_path)
            else:
                raise Exception(
                    "PARAMETER missing  or invalid : the data path for VALIDATION is not valid"
                )
        case "predict":
            is_cuda_enables()
            _data_path = get_parameter(args, "data_path")
            _weights = get_weight_path(args)
            _source = get_parameter(args, "source")
            if _weights and os.path.exists(_weights):
                model = YOLO(model=_weights)
            else:
                raise Exception(
                    "PARAMETER missing  or invalid : the weights path for PREDICTION is not valid"
                )
            if _source and os.path.exists(_source):
                predictions = model.predict(source=_source)
            elif os.path.exists(_data_path):
                predictions = model.predict(source=_data_path)
            # LoggerUtility.log_message(
            #     f"predictions results : {predictions[0]} ", LOG_LEVEL("INFO")
            # )

# experiment_name = get_next_experiment_name()
# model.val()
# model = YOLO("yolov8n.yaml")

# model = YOLO("yolov10n.yaml")
# model.train(data="./resources/dataset/data.yaml", epochs=2)
# model.val(data="./resources/dataset/data.yaml")

# py main.py train --epochs=3 --weight-path=best
# py main.py validate --weight-path=best
# py main.py predict --weight-path=best
# py main.py train --epochs=200 --weight-path=best
# py main.py predict --weight-path=best --source=".\resources\dataset\test\images"
# py main.py predict --weight-path=best --source="d:\projects\AI\deep-learning-class\project2\coins\resources\dataset\test\images"


# py main.py --help-command=prepare-dataset


# path: D:\projects\AI\deep-learning-class\project2\coins\resources\dataset
# path: /d/projects/AI/deep-learning-class/project2/coins/resources/datasets
# names:
#   0: 01-Shekel-Coin
#   1: 10-Shekel-Coin
#   2: 02-Shekel-Coin
#   3: 05-Shekel-Coin
