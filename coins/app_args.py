import argparse
from utils.logging_utils import LOG_LEVEL
import const
from utils.logging_utils import LoggerUtility, LOG_LEVEL
from utils.model_utils import find_most_recent_pt_file
import os


def get_args_help(arg_type):
    msg = ""
    header = f"app parameters for"
    # fmt: off
    match arg_type:
        case "prepare-dataset":
            msg = f"{arg_type} : \n" \
            f"----------------------------------------- \n" \
            f"  --path_to_base   (default : {const.RELATIVE_DATASET_BASE_PATH}) - path to base dataset used as a blueprint for execution \n" \
            f"  --path_to_dataset  (default : .{const.RELATIVE_DATASET_PATH}/data.yaml ) - path to dispatch new dataset to be executed on \n" \
            f"  --use_pct   (default {const.DATASET_USE_PCT} ) - portion of base dataset to use \n" \
            f"  --train_pct (default {const.TRAIN_PCT}) - % of dataset prepared for training \n" \
            f"  --valid_pct (default {const.VALIDATION_PCT}) - % of dataset prepared for validation \n"
            
        case "train":
            msg = f"{arg_type} : \n" \
            f"----------------------------------------- \n" \
            f"  --batch_size  (default {const.BATCH} ) - batch size for each iteration \n" \
            f"  --learning_rate (default {const.LEARNING_RATE} ) - learning rate for training -need to be as small as possible \n" \
            f"  --epochs (default {const.EPOCHS} ) - # epochs to execute   \n" \
            f"  --weights(default=best) - (none/best/last/<experiment version number>/[path]) path to previous trainings weights files \n" \
            f"  --data_path (default .{const.RELATIVE_DATASET_PATH}/data.yaml ) - path to data yaml execution file  \n"

            
        case "validate":
            msg = f"{arg_type} : \n" \
            f"----------------------------------------- \n" \
            f"  --weights (required)  -  best/last/[path]/<experiment id> -  path to previous trainings weights files \n" \
            f"  --data_path  (default {const.DATASET_PATH} )   -  path to dataset for validation , where data-config-yaml is available \n"
            

            
        case "predict":
            msg = f"{arg_type}  : \n" \
            f"----------------------------------------- \n" \
            f"  --weights    (required) --  best/last/[path] -  path to previous trainings weights files \n" \
            f"  --data_path  (default {const.DATASET_PATH} )   - path to data yaml execution file  \n" \
            f"  --source (or data_path is used as config) - path to file / folder to be predicted \n" 
    # fmt: on
    return {"header": header, "msg": msg}


# taking care of config mechanism priority :
# 1 - command line arguments
# 2 - environment variables
def get_parameter(args, parameter_name):
    LoggerUtility.log_message(
        "main - app parameters",
        f" parameter : {parameter_name}  ---> {getattr(args, parameter_name)}",
        LOG_LEVEL["INFO"],
    )
    return getattr(args, parameter_name)


def init_args():
    # load the environment variables file ./resources/.env
    # load_dotenv("./resources/.env")
    # setting command line parser
    parser = argparse.ArgumentParser(
        description="main parser for application", prog="main"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parser.add_argument(
        "--help-command", type=str, help="Show help for a specific command"
    )
    # setting command line arguments
    # fmt: off
    prepare_ds = subparsers.add_parser("prepare-dataset", help="prepare dataset for training,validation and testing chunks")
    prepare_ds.add_argument("--path_to_base"    , type=str , default=const.RELATIVE_DATASET_BASE_PATH, help="path to base dataset used as a blueprint for execution")  
    prepare_ds.add_argument("--path_to_dataset" , type=str , default=const.RELATIVE_DATASET_PATH     , help="path to dispatch new dataset to be executed on")  
    prepare_ds.add_argument("--use_pct"         , type=int ,default=const.DATASET_USE_PCT       , help = "portion of base dataset to use")
    prepare_ds.add_argument("--train_pct"       , type=int ,default=const.TRAIN_PCT             , help = "% of dataset prepared for training")  
    prepare_ds.add_argument("--valid_pct"       , type=int ,default=const.VALIDATION_PCT        , help = "% of dataset prepared for validation")  

    train = subparsers.add_parser("train"  , help="train over the selected dataset")
    train.add_argument("--batch_size"      , type=int ,default=const.BATCH               , help = " batch size for each iteration")
    train.add_argument("--learning_rate"   , type=float ,default=const.LEARNING_RATE       , help = " learning rate for training -need to be as small as possible")  
    train.add_argument("--epochs"          , type=int ,default=const.EPOCHS              , help = " # epochs to execute")  
    train.add_argument("--data_path"       , type=str ,default=const.DATASET_PATH        , help = " path to data yaml execution file " )
    train.add_argument("--weights"    , type=str , default="best" , help = " path to previous trainings weights files (none/best/las/[path]) ")  

    validate = subparsers.add_parser("validate", help="validate the already trained model with the validation chunk only, \n" 
                                      "without setting weights , no back propagations")
    validate.add_argument   ("--data_path"       , type=str ,default=const.DATASET_PATH        , help = " path to data yaml execution file " )
    validate.add_argument("--weights"    , type=str ,default="best"  , help = "path to previous trainings weights files")  
    
    
        
    predict = subparsers.add_parser("predict", help="predict specific path to a file or full directory")
    predict.add_argument  ("--data_path"  , type=str ,default=const.DATASET_PATH        , help = " path to data yaml execution file " )
    predict.add_argument  ("--weights"    , type=str ,default="best"  ,  help = " path to previous trainings weights files (none/best/las/[path])")
    predict.add_argument("--source" , type=str , help = "path to file / folder to be predicted" )  

    
    args, unknown_args = parser.parse_known_args()
    # args = parser.parse_args()
    if args.help_command:
        help_content = get_args_help(args.help_command)

        print(f"{help_content['header']}", f"{help_content['msg']}")
    return args


#  if weight is "best"  or "last" or the <execution number>
#  search for last successful weight
def get_weight_path(args):
    value = get_parameter(args, "weights")
    if value.isnumeric() or (value.lower() in ["best", "last"]):
        return find_most_recent_pt_file(value.lower())
    match value.lower():
        case "none":
            return "yolov8n.pt"
        case _:
            if os.path.exists(value):
                return value
            else:
                return "yolov8n.pt"
