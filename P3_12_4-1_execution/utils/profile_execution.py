import os
import random
import numpy as np
import app_const as const
from dotenv import dotenv_values


from utils.plot_utils import plot_training_history
from utils.dataset_utils import load_dataset
from utils.model_utils import (
    create_model,
    train_model,
    evaluate_model,
    set_model_seed,
)

from utils.timing_utils import capture_and_time
from prettytable import PrettyTable

import tensorflow as tf


def print_execution_summery(
    profile, history, test_loss, test_accuracy, train_execution_time
):
    history_dict = history.history
    # Extract the last epoch's metrics from the history
    val_accuracy = history_dict["val_accuracy"][-1]
    val_loss = history_dict["val_loss"][-1]

    table = PrettyTable()
    # Print the organized table
    table.field_names = [
        "Validation",
        "Test",
        "Inference",
    ]
    sub_headers = [
        "acc | loss",
        "acc | loss",
        "Time (sec)",
    ]
    table.add_row(sub_headers)
    table.add_row(["-" * 22, "-" * 22, "_" * 10])

    table.add_row(
        [
            f"{val_accuracy:.4f} | {val_loss:.4f}",
            f"{test_accuracy:.4f} | {test_loss:.4f}",
            f"{train_execution_time:.4f} sec",
        ]
    )

    print(" ===================================")

    print(" ===================================")
    print(" execution summery for profile : ", profile["NAME"])
    print(" ===================================")

    print(table)

    print(" validation split : ", profile["TRAIN_VALIDATION_SPLIT"])
    print(" reduce factor    : ", profile["TRAIN_DECREASE_FACTOR"])
    print(" number of epocs  : ", profile["TRAIN_EPOCHS"])
    print(" batch size       : ", profile["BATCH_SIZE"])
    print(" learning rate    : ", profile["LEARNING_RATE"])
    print(" is freeze model  : ", profile["FREEZE_MODEL"])
    print(" loss function    : ", profile["LOSS_FUNCTION"])
    print(" dataset name     : ", profile["DATASET_NAME"])
    print(" random seed      : ", profile["SEED"])

    return


# Set the random seeds - if no seed is provided,
# the default value from app_const.py is used
def set_seed(seed=const.SEED):
    # Set the random seeds
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # This variable influences the hash function's behavior in Python 3.3 and later.
    random.seed(seed)
    np.random.seed(seed)
    set_model_seed(seed)
    return


def execute_profile(device_name, profile):

    set_seed(profile["SEED"])
    # Load the dataset
    train_images, train_labels, val_images, val_labels, test_images, test_labels = (
        load_dataset(
            dataset_name=profile["DATASET_NAME"],
            validation_split=profile["TRAIN_VALIDATION_SPLIT"],
            dec_factor=profile["TRAIN_DECREASE_FACTOR"],
        )
    )

    with tf.device(device_name):
        model, model_execution_time = capture_and_time(
            func=create_model,
            train_images=train_images,
            image_size=profile["IMAGE_SIZE"],
        )
        print(f"Model creation time: { model_execution_time} seconds")

        # Do the actual training
        history, model_train_execution_time = capture_and_time(
            func=train_model,
            model=model,
            train_data=(train_images, train_labels),
            validation_data=(val_images, val_labels),
            epochs=profile["TRAIN_EPOCHS"],
            batch_size=profile["BATCH_SIZE"],
        )

        print(f"Model train time: { model_train_execution_time} seconds")

        # Evaluate
        # test_loss, test_acc = evaluate_model(

        (test_loss, test_acc), evaluation_execution_time = capture_and_time(
            func=evaluate_model,
            model=model,
            test_data=(test_images, test_labels),
            verbose=const.VERBOSE[profile["VERBOSE_LOG_DETAILS"]],
        )

        # test_loss = test_result[0]
        # test_acc = test_result[1]

        print(f"Model evaluate time: { evaluation_execution_time} seconds")
        print(f"Model evaluate loss: { test_loss } accuracy: { test_acc }")

        print_execution_summery(
            profile=profile,
            test_accuracy=test_acc,
            test_loss=test_loss,
            history=history,
            train_execution_time=model_train_execution_time,
        )

        plot_training_history(history=history)
