from utils.logging_utils import LoggerUtility, LOG_LEVEL
from collections import Counter
from app_args import get_parameter, get_weight_path
from utils.model_utils import is_cuda_enables
from ultralytics import YOLO
from utils.dataset_utils import get_source_from_data_yaml
from const import coins_values, labels_values
from utils.plot_utils import compare_labeled_and_prediction
import os


def get_prediction_summary(predictions):
    predictions_summary = []
    # Retrieve the unique class names based on class IDs
    class_names = predictions[0].names
    # Iterate over each result (each image)
    for prediction in predictions:
        path = prediction.path
        # get the name of the image file without path and .jpg
        image_name = prediction.path.split("\\")[-1]
        # Get class IDs for the predicted objects
        class_ids = prediction.boxes.cls.cpu().numpy()  # Convert tensor to numpy array

        # Count occurrences of each class
        class_counts = Counter(class_ids)

        unique_labels = [
            prediction.names[int(class_id)] for class_id in class_counts.keys()
        ]
        # Create the output dictionary for this image
        image_summary = {"file_name": image_name, "coins": {}}

        # Add the count of each class name
        for class_id, count in class_counts.items():
            class_name = class_names[
                int(class_id)
            ]  # Get the class name from the class ID
            image_summary["coins"][class_name] = count

        # Append the summary for this image to the results list
        predictions_summary.append(image_summary)

    return predictions_summary


def calculate_predictions_value(predictions_object):
    total_value = 0
    summery = []
    for prediction in predictions_object:
        coins = prediction["coins"]
        if len(coins) == 0:
            summery.append({"file": prediction["file_name"], "value": 0})
            continue
        prediction_value = 0
        for coin, count in coins.items():
            for coin_value in coins_values:
                if coin in coin_value:
                    prediction_value += coin_value[coin] * count
        summery.append({"file": prediction["file_name"], "value": prediction_value})
        total_value += prediction_value
    summery.append({"total_value": total_value})
    return summery


def calculate_labeled_data_summary(path):
    summery = []
    total_value = 0
    for filename in os.listdir(path):
        file_value = 0
        file_path = os.path.join(path, filename)
        with open(file_path, "r") as file:
            labeled_data = file.readlines()
            item_counter = Counter()
            for line in labeled_data:
                parts = line.strip().split()
                if parts:
                    item_id = parts[0]
                    item_counter[item_id] += 1

            for item, count in item_counter.items():
                file_value += labels_values[item] * count

        summery.append({"file": filename, "value": file_value})
        total_value += file_value
    summery.append({"total": total_value})
    return summery


def predict(args):
    cuda = is_cuda_enables()
    _data_path = get_parameter(args, "data_path")
    _weights = get_weight_path(args)
    _source = get_parameter(args, "source")
    labels_path = ""
    if _weights and os.path.exists(_weights):
        model = YOLO(model=_weights)
    else:
        raise Exception(
            "PARAMETER missing  or invalid : the weights path for PREDICTION is not valid"
        )
        # if source is null or not valid pathe and data path exists
    if _source != None and os.path.exists(_source):
        pass
    elif _data_path and os.path.exists(os.path.abspath(_data_path)):
        try:
            path = get_source_from_data_yaml("predict", _data_path)
            labels_path = f"{path}/labels"
            _source = f"{path}/images"
        except Exception as e:
            LoggerUtility.log_message(
                f"error while trying to get prediction path from data yaml : {e} ",
                LOG_LEVEL("ERROR"),
            )
    try:
        predictions = model.predict(source=_source)
        predictions_obj = get_prediction_summary(predictions)
        prediction_summery = calculate_predictions_value(predictions_obj)
        print(f"predictions summery: {prediction_summery} ")
        if labels_path and os.path.exists(labels_path):
            labeled_data_summary = calculate_labeled_data_summary(labels_path)
            print(f"labeled summery: {labeled_data_summary} ")
            compare_labeled_and_prediction(labeled_data_summary, prediction_summery)

        total_value = prediction_summery[-1]["total_value"]
        print(" \n =============================================================")
        print(" \n ============P R E D I C T I O N -- S U M M E R Y ============")
        print(" \n =============================================================")
        print("     Total Value:        ", total_value)
        print(" \n =============================================================")
        print(" \n =============================================================")
        print(" \n =============================================================")

    except Exception as e:
        LoggerUtility.log_message(
            f"error while trying to predict : {e} ", LOG_LEVEL("ERROR")
        )


# [
#     {
#         "file_name": "f6484ad0-ido_50_jpg.rf.8c0cc55ae03f18ffdbe475914ce334fb",
#         "coins": {"10-Shekel-Coin": 1},
#     },
#     {
#         "file_name": "f98167b6-IMG_3011_JPG.rf.9eae9bd30c7d582a4cf13ae9ba753c0d",
#         "coins": {"01-Shekel-Coin": 2, "05-Shekel-Coin": 1},
#     },
#     {
#         "file_name": "fb773ecb-IMG_7479_jpg.rf.259721f9c346148f751c990434e264d2",
#         "coins": {},
#     },
# ]
