import utils.comet_utils as COMET
from ultralytics import YOLO
import os
import numpy as np
from datetime import datetime


import const
from utils.model_utils import is_cuda_enables
from utils.yolo_dataset_utils import prepare_dataset, get_source_from_data_yaml
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

    if command == "help":
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
            cuda = is_cuda_enables()
            _data_path = get_parameter(args, "data_path")
            _weights = get_weight_path(args)
            _source = get_parameter(args, "source")
            if _weights and os.path.exists(_weights):
                model = YOLO(model=_weights)
            else:
                raise Exception(
                    "PARAMETER missing  or invalid : the weights path for PREDICTION is not valid"
                )
                # if source is null or not valid pathe and data path exists
            if (_source != None and os.path.exists(_source)) or os.path.exists(
                os.path.abspath(_data_path)
            ):
                try:
                    _source = get_source_from_data_yaml(command, _data_path)
                except Exception as e:
                    LoggerUtility.log_message(
                        f"error while trying to get prediction path from data yaml : {e} ",
                        LOG_LEVEL("ERROR"),
                    )
            try:
                predictions = model.predict(source=_source)

                # for prediction in predictions:
                #     LoggerUtility.log_message(
                #         f"predictions results : {parse_prediction(prediction)} ",
                #         LOG_LEVEL("INFO"),
                #     )
            except Exception as e:
                LoggerUtility.log_message(
                    f"error while trying to predict : {e} ", LOG_LEVEL("ERROR")
                )
