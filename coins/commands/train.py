from utils.logging_utils import LoggerUtility, LOG_LEVEL
from app_args import get_parameter, get_weight_path
from utils.model_utils import verify_cuda_enabled
from ultralytics import YOLO
import os
from utils.main_utils import get_execution_path
import const


def train(experiment, args):
    verify_cuda_enabled()
    execution_path = get_execution_path(args=args, command="train")
    _weights = get_weight_path(args)
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
        + f"_weights : {_weights}  ",
        LOG_LEVEL["INFO"],
    )

    if _weights and os.path.exists(_weights):
        model = YOLO(model=_weights)
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
            pretrained=True,
        )
    else:
        raise Exception(
            "PARAMETER missing  or invalid : the data path for TRAIN is not valid"
        )
