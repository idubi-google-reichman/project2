import os
import const
from app_args import init_args, get_parameter, get_weight_path
from utils.logging_utils import LoggerUtility, LOG_LEVEL
import stat


def set_execution_path(relative_path, command):
    fullpath = os.path.join(relative_path, command)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
        os.chmod(fullpath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return f"{fullpath}/"


def get_execution_path(args, command):
    if command == "prepare-dataset":
        execution_path = get_parameter(args, "path_to_dataset")
    else:
        execution_path = set_execution_path(const.EXECUTION_PATH, command)
    return execution_path
