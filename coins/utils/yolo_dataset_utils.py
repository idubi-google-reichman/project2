import os
import shutil
import random
import yaml
from utils.logging_utils import LoggerUtility, LOG_LEVEL


def prepare_dataset(
    path_to_base,
    path_to_dataset,
    use_pct,
    train_pct=70,
    valid_pct=15,
):
    """
    Prepare a dataset for machine learning by splitting it into training, validation, and testing sets.
    test_pct  - will be calculated out of 100% - train - validate

    Parameters:
        path_to_base (str): Path to the base directory containing images and labels.
        path_to_dataset (str): Path to the dataset directory.
        train_pct (float, optional): Percentage of files to use for training. Defaults to 0.
        valid_pct (float, optional): Percentage of files to use for validation. Defaults to 0.
    """

    # Step 1: validate percentage values
    if valid_pct < 0 or train_pct < 0:
        raise ValueError("train ration cant be negative")
    if (train_pct + valid_pct) > 100:
        LoggerUtility.log_message(
            f"dataset-utils",
            f" prepare_dataset - ERROR : Sum of percentages values is greater than 100 : {train_pct}\%  , validation: {valid_pct}\% ",
            LOG_LEVEL["ERROR"],
        )

        raise ValueError("Sum of percentages values is greater than 100")

    test_pct = 100 - (train_pct + valid_pct)

    LoggerUtility.log_message(
        f"dataset-utils",
        f" prepare_dataset - preparing dataset for training  train : {train_pct}\%  , validation: {valid_pct}\%  test: {test_pct}\%  ",
        LOG_LEVEL["INFO"],
    )

    # Step 2: Count files in base/images and base/labels
    path_to_base = os.path.join(os.getcwd(), os.path.normpath(path_to_base))
    path_to_dataset = os.path.join(os.getcwd(), os.path.normpath(path_to_dataset))
    images_path = os.path.join(os.path.normpath(path_to_base), "images")
    labels_path = os.path.join(os.path.normpath(path_to_base), "labels")
    #  get all images names
    images_files = [
        f
        for f in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, f))
    ]
    # get all notes names
    labels_files = [
        f
        for f in os.listdir(labels_path)
        if os.path.isfile(os.path.join(labels_path, f))
    ]

    # Step 2: Ensure the count of images and labels is the same
    assert len(images_files) == len(
        labels_files
    ), "Mismatch between images and labels count"

    # Step 3: Delete all folders under path_to_dataset
    if os.path.exists(path_to_dataset):
        for folder in os.listdir(path_to_dataset):
            folder_path = os.path.join(path_to_dataset, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
    LoggerUtility.log_message(
        f"dataset-utils",
        f" deleted dataset at {path_to_dataset} , prepare for new dataset from template  ",
        LOG_LEVEL["INFO"],
    )

    # Create directories for train, validation, and test
    train_images_path = os.path.join(path_to_dataset, "train/images")
    train_labels_path = os.path.join(path_to_dataset, "train/labels")
    valid_images_path = os.path.join(path_to_dataset, "valid/images")
    valid_labels_path = os.path.join(path_to_dataset, "valid/labels")
    test_images_path = os.path.join(path_to_dataset, "test/images")
    test_labels_path = os.path.join(path_to_dataset, "test/labels")

    os.chmod(path_to_dataset, mode=755)
    os.makedirs(train_images_path, mode=755)
    os.makedirs(train_labels_path, mode=755)
    os.makedirs(valid_images_path, mode=755)
    os.makedirs(valid_labels_path, mode=755)
    os.makedirs(test_images_path, mode=755)
    os.makedirs(test_labels_path, mode=755)

    # Step 4: Create unique arrays for validation and test file names
    # use pct is used to work on small dataset for develop purposes
    use_pct = use_pct / 100

    total_files = int(len(images_files) * use_pct)
    indices = list(range(total_files))
    random.shuffle(indices)

    valid_pct = valid_pct / 100
    train_pct = train_pct / 100
    test_pct = test_pct / 100

    valid_count = int(total_files * valid_pct)
    test_count = int(total_files * test_pct)

    valid_indices = indices[:valid_count]
    test_indices = indices[valid_count : valid_count + test_count]
    train_indices = indices[valid_count + test_count :]

    # Step 5: Copy files to respective directories
    def copy_files(
        indices, src_images_path, src_labels_path, dst_images_path, dst_labels_path
    ):
        for i in indices:
            image_file = images_files[i]
            label_file = labels_files[i]

            # Ensure same file name
            assert (
                image_file[:-4] == label_file[:-4]
            ), "Mismatched image and label file names"

            shutil.copy(
                os.path.join(src_images_path, image_file),
                os.path.join(dst_images_path, image_file),
            )
            shutil.copy(
                os.path.join(src_labels_path, label_file),
                os.path.join(dst_labels_path, label_file),
            )

    def copy_meta_files(path_to_base, path_to_dataset):
        shutil.copyfile(
            os.path.join(path_to_base, "data.yaml"),
            os.path.join(path_to_dataset, "data.yaml"),
        )
        shutil.copyfile(
            os.path.join(path_to_base, "classes.txt"),
            os.path.join(path_to_dataset, "classes.txt"),
        )
        shutil.copyfile(
            os.path.join(path_to_base, "data.yaml"),
            os.path.join(path_to_dataset, "data.yaml"),
        )

    def prepare_YOLO_config(path_to_dataset):
        with open(os.path.join(path_to_dataset, "data.yaml"), "r") as file:
            data = yaml.safe_load(file)

        # Update the values
        data["path"] = path_to_dataset
        # data["train"] = os.path.join(path_to_dataset, "train")
        # data["val"] = os.path.join(path_to_dataset, "valid")
        # data["test"] = os.path.join(path_to_dataset, "test")
        data["train"] = "./train"
        data["val"] = "./valid"
        data["test"] = "./test"

        # Write the updated data back to the YAML file
        with open(os.path.join(path_to_dataset, "data.yaml"), "w") as file:
            yaml.safe_dump(data, file)

    path_to_base = os.path.join(os.getcwd(), os.path.normpath(path_to_base))
    path_to_dataset

    copy_files(
        train_indices, images_path, labels_path, train_images_path, train_labels_path
    )
    copy_files(
        valid_indices, images_path, labels_path, valid_images_path, valid_labels_path
    )
    copy_files(
        test_indices, images_path, labels_path, test_images_path, test_labels_path
    )

    LoggerUtility.log_message(
        f"dataset-utils",
        f" copied files from base dataset:{path_to_base}  , to dataset: {path_to_dataset} ",
        LOG_LEVEL["INFO"],
    )

    copy_meta_files(path_to_base, path_to_dataset)

    prepare_YOLO_config(
        path_to_dataset,
    )


def get_source_from_data_yaml(command, _data_path):
    with open(_data_path, "r") as file:
        data = yaml.safe_load(file)
    path = data["path"]

    match command:
        case "predict":
            path = f"{path}/{data['test']}"
        case "validate":
            path = f"{path}/{data['val']}"
        case "train":
            path = f"{path}/{data['train']}"
        case _:
            path = f"{path}/{data['train']}"

    if os.path.exists(path):
        return path
    else:
        raise Exception("Invalid path in data.yaml file")


# # Example usage:
# prepare_dataset(
#     path_to_base="./resources/base_dataset",
#     path_to_dataset="./resources/dataset",
#     use_pct=100,
#     train_pct=70,
#     valid_pct=15,
#     test_pct=15,
# )
