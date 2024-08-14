import utils.comet_utils as COMET

from ultralytics import YOLO

import os

from datetime import datetime
import stat

import const
from utils.model_utils import verify_cuda_enabled
from utils.yolo_dataset_utils import prepare_dataset
from utils.logging_utils import LoggerUtility, LOG_LEVEL
from utils.plot_utils import plot_and_log_curves, plot_confusion_matrix

from app_args import init_args, get_parameter, get_weight_path


# exit()
# Create a new YOLO model from scratch


def set_execution_path(base_path, command):
    fullpath = os.path.join(base_path, command)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
        os.chmod(fullpath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return f"{fullpath}/"


comet_config = COMET.get_comet_configuration()
if __name__ == "__main__":
    args = init_args()
    if args.help_command:
        LoggerUtility.log_message("main - args", "")
        exit()

    command = get_parameter(args, "command")

    if (not command) or (command not in const.ALL_EXECUTION_MODES):
        print(f"none of the execution modes {const.ALL_EXECUTION_MODES} is recognized ")
        exit(-1)
    if command == "prepare-dataset":
        execution_path = get_parameter(args, "path_to_dataset")
    else:
        execution_path = set_execution_path(const.EXECUTION_PATH, command)
        experiment = COMET.create_experiment(comet_config=comet_config)
        _weights_path = get_weight_path(args)

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
            verify_cuda_enabled()
            _epochs = get_parameter(args, "epochs")
            _learning_rate = get_parameter(args, "learning_rate")
            _batch_size = get_parameter(args, "batch_size")
            _data_path = get_parameter(args, "data_path")
            LoggerUtility.log_message(
                "model train ->  \n ",
                f"_epochs : {_epochs} /n "
                + f"_learning_rate : {_learning_rate} /n "
                + f"_batch_size : {_batch_size} /n "
                + f"_data_path : {_data_path} /n "
                + f"_weights_path : {_weights_path}  ",
                LOG_LEVEL["INFO"],
            )

            if _weights_path and os.path.exists(_weights_path):
                model = YOLO(model=_weights_path)
                for param in model.parameters():
                    param.requires_grad = True
            else:
                model = YOLO(model="yolov8n.yaml")

            if os.path.exists(_data_path):
                model.train(
                    data=_data_path,
                    epochs=_epochs,
                    lr0=_learning_rate,
                    batch=_batch_size,
                    project=execution_path,
                    name=experiment.name,
                )
            else:
                raise Exception(
                    "PARAMETER missing  or invalid : the data path for TRAIN is not valid"
                )
        case "validate":
            verify_cuda_enabled()
            _data_path = get_parameter(args, "data_path")

            if _weights_path and os.path.exists(_weights_path):
                model = YOLO(model=_weights_path)

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
            verify_cuda_enabled()
            _data_path = get_parameter(args, "data_path")
            _prediction_path = get_parameter(args, "prediction_path")
            if _weights_path and os.path.exists(_weights_path):
                model = YOLO(model=_weights_path)
            else:
                raise Exception(
                    "PARAMETER missing  or invalid : the weights path for PREDICTION is not valid"
                )
            if _prediction_path and os.path.exists(_prediction_path):
                predictions = model.predict(source=_prediction_path)
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
# py main.py predict --weight-path=best --prediction_path=".\resources\dataset\test\images"
# py main.py predict --weight-path=best --prediction_path="d:\projects\AI\deep-learning-class\project2\coins\resources\dataset\test\images"


# py main.py --help-command=prepare-dataset
