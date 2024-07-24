# all profiles from class assignment are here
# starting with the base profile (as we saw in the class example)
# changing as we progress to 3A...3E accumulating the changes in the profile
PROFILES = {
    "BASE_PROFILE": {
        "NAME": "BASE",
        "MODEL_NAME": "MobileNet",
        "DATASET_NAME": "cifar10",
        "TRAIN_VALIDATION_SPLIT": 0.2,
        "SEED": 42,
        "TRAIN_DECREASE_FACTOR": 80,
        "TRAIN_EPOCHS": 1,
        "BATCH_SIZE": 32,
        "IMAGE_SIZE": 224,
        "LOSS_FUNCTION": "categorical_crossentropy",
        "LEARNING_RATE": 0.001,
        "FREEZE_MODEL": True,
        "VERBOSE_LOG_DETAILS": "SINGLE_LINE",
        "OPTIMIZER_NAME": "Adam",
        "MODEL_CONFIG": {
            "input_shape": (32, 32, 3),
            "layers": [
                {
                    "type": "Resizing",
                    "params": {
                        "height": 224,
                        "width": 224,
                        "interpolation": "nearest",
                        "input_shape": ":resize_train_images_shape:",
                    },
                },
                {"type": "BaseModel", "params": {}},
                {"type": "GlobalAveragePooling2D", "params": {}},
                {"type": "Dense", "params": {"units": 10, "activation": "softmax"}},
            ],
        },
    },
    "3": {
        "NAME": "3",
        "TRAIN_EPOCHS": 5,
        "TRAIN_DECREASE_FACTOR": 10,
    },
    "3.A": {
        "NAME": "3.A",
        "FREEZE_MODEL": False,
    },
    "3.B": {"NAME": "3.B", "LEARNING_RATE": 0.01},
    "3.C": {
        "NAME": "3.C",
        "MODEL_CONFIG": {
            "input_shape": (32, 32, 3),
            "layers": [
                {
                    "type": "Resizing",
                    "params": {
                        "height": 224,
                        "width": 224,
                        "interpolation": "nearest",
                        "input_shape": ":resize_train_images_shape:",
                    },
                },
                {"type": "BaseModel", "params": {}},
                {"type": "GlobalAveragePooling2D", "params": {}},
                {"type": "Dropout", "params": {"rate": 0.5}},
                {"type": "Dense", "params": {"units": 10, "activation": "softmax"}},
            ],
        },
    },
    "3.D": {
        "NAME": "3.D",
        "MODEL_CONFIG": {
            "input_shape": (32, 32, 3),
            "layers": [
                {
                    "type": "Resizing",
                    "params": {
                        "height": 224,
                        "width": 224,
                        "interpolation": "nearest",
                        "input_shape": ":resize_train_images_shape:",
                    },
                },
                {"type": "BaseModel", "params": {}},
                {"type": "RandomFlip", "params": {"mode": "horizontal_and_vertical"}},
                {"type": "GlobalAveragePooling2D", "params": {}},
                {"type": "Dropout", "params": {"rate": 0.5}},
                {"type": "Dense", "params": {"units": 10, "activation": "softmax"}},
            ],
        },
    },
    "3.E": {
        "NAME": "3.E",
        "MODEL_NAME": "Resnet",
        "SUB_MODEL_NAME": "18",
    },
}
