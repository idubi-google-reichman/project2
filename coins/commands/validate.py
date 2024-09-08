from app_args import get_parameter, get_weight_path
from ultralytics import YOLO
from utils.model_utils import is_cuda_enables
import os
import utils.comet_utils as COMET


def validate(experiment, args):
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
