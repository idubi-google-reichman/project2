from app_args import init_args, get_parameter, get_weight_path
from commands.predict import predict
from commands.prepare_dataset import prepare_dataset
from commands.train import train
from commands.validate import validate
from datetime import datetime
from ultralytics import YOLO
from utils.logging_utils import LoggerUtility, LOG_LEVEL
from utils.main_utils import get_execution_path
from utils.model_utils import is_cuda_enables
from const import ALL_EXECUTION_MODES
import numpy as np
import os
import utils.comet_utils as COMET


if __name__ == "__main__":
    args = init_args()
    if args.help_command:
        LoggerUtility.log_message("main - args", "")
        exit()

    command = get_parameter(args, "command")

    if (not command) or (command not in ALL_EXECUTION_MODES):
        print(f"none of the execution modes {ALL_EXECUTION_MODES} is recognized ")
        exit(-1)

    if command not in ["prepare-dataset", "predict"]:
        # create experiment that will be managed by COMIT
        experiment = COMET.create_experiment()

    execution_path = get_execution_path(args=args, command=command)

    LoggerUtility.configure_logger(
        log_level=LOG_LEVEL["INFO"],
        log_file_path=f"{execution_path}",
    )

    match command:
        case "prepare-dataset":
            prepare_dataset(args=args)
        case "train":
            train(experiment=experiment, args=args)
        case "validate":
            validate(experiment=experiment, args=args)
        case "predict":
            predict(args=args)
