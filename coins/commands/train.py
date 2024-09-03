from utils.logging_utils import LoggerUtility, LOG_LEVEL
from app_args import get_parameter, get_weight_path
from utils.model_utils import is_cuda_enables
from ultralytics import YOLO

from utils.main_utils import get_execution_path
import const
import os


def train(experiment, args):
    cuda = is_cuda_enables()
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

    model = YOLO(model=_weights)
    for param in model.parameters():
        param.requires_grad = True

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
