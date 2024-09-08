import os
import yaml


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
        return f"{path}"
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
