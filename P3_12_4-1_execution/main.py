import os
import random
import numpy as np
import app_const as const
import tensorflow as tf
from utils.plot_utils import (
    plot_training_history,
    print_training_report,
)
from utils.dataset_utils import load_dataset
from utils.model_utils import (
    create_model,
    train_model,
    evaluate_model,
    set_seed as set_model_seed,
)
from utils.timing_utils import capture_and_time
from sklearn.linear_model import LogisticRegression


# Set the random seeds
def set_seed(seed=42):
    # Set the random seeds
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # This variable influences the hash function's behavior in Python 3.3 and later.
    random.seed(seed)
    np.random.seed(seed)
    set_model_seed(42)
    return


set_seed(42)

# Load the dataset
train_images, train_labels, val_images, val_labels, test_images, test_labels = (
    load_dataset(
        # Load the dataset dynamically using getattr
        dataset_name=const.DATASET_NAME,
        validation_split=const.TRAIN_VALIDATION_SPLIT,
        dec_factor=const.TRAIN_DECREASE_FACTOR,
    )
)


# Create the backbone model that will be used to train
with tf.device("/GPU:0"):
    model, model_execution_time = capture_and_time(
        create_model, train_images=train_images, image_size=const.IMAGE_SIZE
    )
    print(f"Model creation time: { model_execution_time} seconds")

    # Do the actual training
    history, model_train_execution_time = capture_and_time(
        train_model,
        model=model,
        train_data=(train_images, train_labels),
        validation_data=(val_images, val_labels),
        epochs=const.TRAIN_EPOCHS,
        batch_size=const.BATCH_SIZE,
    )

    print(f"Model train time: { model_train_execution_time} seconds")

    # Evaluate
    # training_stats , evaluation_execution_time=
    test_loss, test_acc = evaluate_model(
        model=model,
        test_data=(test_images, test_labels),
        log_detailed_level=const.LOG_DETAILS["VERBOS"],
    )

    print_training_report(
        test_accuracy=test_acc,
        test_loss=test_loss,
        history=history,
        train_execution_time=model_train_execution_time,
    )

    plot_training_history(history=history)
