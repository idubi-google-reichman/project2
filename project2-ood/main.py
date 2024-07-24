import os
import textwrap
from datetime import datetime
import app_const as const
import stat
import argparse

# from dotenv import load_dotenv
from entities.ProfileModelFactory import ModelFactory
from utils.dataset_utils import load_dataset
from utils.logging_utils import LoggerUtility, LOG_LEVEL

import json
from prettytable import PrettyTable
import tensorflow as tf
from profiles import PROFILES


# Function to get device pointer
# we search for gpu and if we find it we return it
# if notthen we return the cpu pointer
def get_active_device_name(allow_gpu=True):
    if allow_gpu and tf.config.list_physical_devices("GPU"):
        return "/device:GPU:0"  # Assuming the first GPU is desired
    else:
        return "/device:CPU:0"


# this is the main function that execute the profile
# create a model out of it and test on it with the dataset
def train_model_with_profile_data(device_name, profile_data, save_path=None):

    profile_data["DEVICE_NAME"] = device_name

    train_images, train_labels, val_images, val_labels, test_images, test_labels = (
        load_dataset(
            dataset_name=profile_data["DATASET_NAME"],
            validation_split=profile_data["TRAIN_VALIDATION_SPLIT"],
            dec_factor=profile_data["TRAIN_DECREASE_FACTOR"],
        )
    )

    # we  use a model factory design patern to create the model
    # the create model static function will create the model based on the profile data
    #  - MODEL_NAME --> Resnet or MobileNet
    #  - SUB_MODEL_NAME --> 18   (can be added later more sub models for resnet)

    model = ModelFactory.create_model(
        profile_data=profile_data, train_image_shape_for_resize=train_images.shape[1:]
    )

    # after creating the model we initiate it on creation with constructor
    # then we compile it
    # model_compile_execution_time is the time in human readable format that took to compile the model

    model_compile_execution_time = model.compile_model()

    LoggerUtility.log_message(
        f"profile_{model.name}",
        f"compiling model time : {model_compile_execution_time}",
        LOG_LEVEL["INFO"],
    )

    # after compiling the model
    # we train it
    # model_train_execution_time is the time in human readable format that took to compile the model
    # result of this execution is the training history
    # in any time we can call model.history() to get the history of the last training
    history, model_train_execution_time = model.train_model(
        train_data=(train_images, train_labels),
        validation_data=(val_images, val_labels),
    )

    LoggerUtility.log_message(
        f"profile_{model.name}",
        f"training time : {model_train_execution_time}",
        LOG_LEVEL["INFO"],
    )

    # after training the model we evaluate it on the test dataset to get the test loss and accuracy
    test_loss, test_accuracy, evaluation_time = model.evaluate_model(
        test_data=(test_images, test_labels),
        verbose=model.verbose,
    )

    LoggerUtility.log_message(
        f"profile_{model.name}",
        f"evaluation time : {evaluation_time}",
        LOG_LEVEL["INFO"],
    )

    LoggerUtility.log_message(
        f"profile_{model.name}",
        f"test loss : {test_loss} , test accuracy : {test_accuracy} ",
        LOG_LEVEL["DEBUG"],
    )

    # we show the summary of the model
    # summery is printed to stdout and not to the log file
    # this is since I dont want log level to override this summery by choice

    model.show_summary()

    # save path to plot png file
    save_path = f"{save_path}profile_{model.name}"

    # plot the training epochs graph and save it on disk in the save_path (./resources/date/time/profile_model_name)
    model.plot_training_history(save_path)

    return model


# since I maerje the profiles - I need to merge the jsons inside it if there is
def merge_json(A, B):
    result = B.copy()  # Start with a copy of B to ensure we don't modify the original B
    for key, value in A.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_json(value, result.get(key, {}))
        else:
            result[key] = value
    return result


# all profiles are in the PROFILES.py file as a parameter
# we decide which profile to loa with the execution_set parameter
# since the profiles are accumulating changes while changing (as explained by Andrey on assignment day)
# so even if we dont want to execute it - we need to merge the current profile with previous profile
# and merge it also on previous profile  ....
def merge_profiles(new_profile, base_profile):
    output_profile = base_profile.copy()
    # Start with a copy of base to ensure we don't modify the original base
    for key, value in new_profile.items():
        if isinstance(value, dict) and key in output_profile:
            output_profile[key] = merge_json(value, output_profile.get(key, {}))
        else:
            output_profile[key] = value
    return output_profile


# to print the profile in the is a dictionary
# and print it in a table format
def flatten_dict(dictionary, parent_key="", sep="_"):
    items = []
    for item_name, item_value in dictionary.items():
        new_key = f"{parent_key}{sep}{item_name}" if parent_key else item_name
        if isinstance(item_value, dict):
            items.extend(flatten_dict(item_value, new_key, sep=sep).items())
        else:
            items.append((new_key, item_value))
    return dict(items)


# print the profile in a table format
def print_profile_in_table(profile):

    # Flatten the dictionary
    flattened_data = flatten_dict(profile)

    table = PrettyTable()
    # Set the column names
    table.field_names = ["Key", "Value"]

    # Maximum width for each column
    max_width = 80

    # Add rows to the table
    for key, value in flattened_data.items():
        wrapped_value = "\n - ".join(textwrap.wrap(str(value), max_width))
        table.add_row([key, wrapped_value])

    # Set alignment for all columns to left
    table.align["Key"] = "l"
    table.align["Value"] = "l"

    # Print the table
    LoggerUtility.log_message(
        f'profile_{profile["NAME"]}',
        f" \n{table}",
        LOG_LEVEL["INFO"],
    )


# the plot and logs will be saved here (./resources/<date>/<time>/)
def set_execution_path(base_path):
    now = datetime.now()
    date_str = now.strftime("%d%m%Y")
    time_str = now.strftime("%H%M%S")
    fullpath = f"{base_path}{date_str}/{time_str}/"
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
        os.chmod(fullpath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return fullpath


# the profiles are configured in profiles.py file (PROFILES = {....})  - we can add more profiles to the PROFILES file
# only items in the execution_set will be executed trained models ,
# other will only serve to pass the profile configuration  to the next profile in line
# if execution_set is not defined - all profiles will be executed
def execute_all_profiles(
    device_name, save_path, execution_set={"3", "3.A", "3.B", "3.C", "3.D", "3.E"}
):
    base_profile = PROFILES["BASE_PROFILE"]

    last_profile = base_profile
    for profile_name in PROFILES:
        profile = PROFILES[profile_name]
        last_profile = merge_profiles(profile, last_profile)
        if profile_name in execution_set:
            LoggerUtility.log_message(
                f"profile_{profile_name}",
                f'Executing profile:{last_profile["NAME"]}',
                LOG_LEVEL["INFO"],
            )
            print_profile_in_table(last_profile)
            # ···························································
            # :    EXECUTE THE PROFILE  - create model and test it      :
            # ···························································
            train_model_with_profile_data(
                device_name=device_name,
                profile_data=last_profile,
                save_path=save_path,
            )
            # -----------------------------------------------------------
        else:
            LoggerUtility.log_message(
                f"profile_{profile_name}",
                f'ignoring  profile: {last_profile["NAME"]} since its not in the execution set',
                LOG_LEVEL["INFO"],
            )


# region - taking care of configuration


# load the environment variables file ./resources/.env
# load_dotenv("./resources/.env")
# setting command line parser
parser = argparse.ArgumentParser()


# taking care of config mechanism priority :
# 1 - command line arguments
# 2 - environment variables
# 3 - const default values
def get_parameter(parameter_name):
    return getattr(args, parameter_name)


# setting command line arguments
parser.add_argument(
    "--execution_set",
    default=const.EXECUTION_SET,
    type=str,
    help=f"execution list of profiles to execute (default {const.EXECUTION_SET}",
),
parser.add_argument(
    "--log_level",
    default=const.LOG_LEVEL,
    choices=LOG_LEVEL.keys(),
    type=str,
    help=f"log level to use (either one of : DEBUG, INFO, WARNING, ERROR, CRITICAL) default {const.LOG_LEVEL} ",
),
parser.add_argument(
    "--execution_path",
    default=const.EXECUTION_PATH,
    type=str,
    help=f"path for logs and plots default {const.EXECUTION_PATH}   ",
)
parser.add_argument(
    "--use_gpu",
    default="True",
    type=str,
    choices=["True", "False", "true", "false", "TRUE", "FALSE"],
    help=f"use gpu if available or cpu only (default : True )",
)


args = parser.parse_args()


# endregion - taking care of configuration


execution_path = set_execution_path(get_parameter("execution_path"))
log_level_name = get_parameter("log_level")
allow_gpu = get_parameter("use_gpu").lower() == ("true")

# Get the active device name - if gpu us available or cpu only
device_name = get_active_device_name(allow_gpu=allow_gpu)
# Configure the logger once at the start of your application

LoggerUtility.configure_logger(
    log_level=LOG_LEVEL[log_level_name], log_file_path=f"{execution_path}"
)
# execute the profiles
# examples :
# --------------------------------------------------------------------------------------------
#    - execution_set = {"3", "3.A", "3.B", "3.C", "3.D", "3.E"} --> all this profiles will be executed
#    - execution_set = {"3.C", "3.E"}  --> only this 2 profiles will be executed
#    - execution_set = {"3.E"}  --> only 3.E profile will be executed
# --------------------------------------------------------------------------------------------
execution_set = set(get_parameter("execution_set").split(","))

execute_all_profiles(
    device_name=device_name,
    execution_set=execution_set,
    save_path=f"{execution_path}",
)
