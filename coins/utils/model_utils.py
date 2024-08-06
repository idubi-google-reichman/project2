from utils.comet_utils import get_next_experiment_name
from ultralytics import YOLO
from utils.logging_utils import LoggerUtility, LOG_LEVEL
import glob
import os
import torch


def verify_cuda_enabled():
    if torch.cuda.is_available():
        msg_cuda_enabled = "is"
    else:
        msg_cuda_enabled = "is not"

    print(f"cuda {msg_cuda_enabled} available")


# get the path to last valid best.pt file created in train folder
def find_most_recent_pt_file(pt_file_name="last"):
    # Define the pattern for searching
    pattern = f"./resources/executions/train/experiment*/weights/{pt_file_name}.pt"
    # Get a list of all files matching the pattern
    matching_files = glob.glob(pattern)
    # Check if any files are found
    if not matching_files:
        print(f"No '{pt_file_name}.pt' files found.")
        return None
    # Sort the files by modification time (most recent first)
    most_recent_file = max(matching_files, key=os.path.getmtime)
    # Return the most recent file path
    return most_recent_file


# Find the most recent 'best.pt' file
# most_recent_best_pt = find_most_recent_pt()
# if most_recent_best_pt:
#     print(f"The most recent 'best.pt' file is located at: {most_recent_best_pt}")


def save_model_stats(model_evaluation, experiment):

    # Confusion Matrix
    confusion_matrix = model_evaluation.confusion_matrix

    # Precision, Recall, F1 Score, and PR Curve
    # precision = model_evaluation.precision  # or results['precision']
    # recall = model_evaluation.recall  # or results['recall']
    # f1 = model_evaluation.f1  # or results['f1']
    # pr_curve = model_evaluation.pr_curve  # or results['pr_curve']
    LoggerUtility.log_message(
        f"model-utility - logging model evaluation (experiment {experiment.name}) :",
        f"confusion matrix :  {confusion_matrix }  \n  ",
        LOG_LEVEL["INFO"],
    )
