from entities.Profile import Profile
from tensorflow.keras.applications import MobileNet


class MobileNetModel(Profile):

    def __init__(self, profile_data):
        self.set_base_model()
        super().__init__(profile_data)

    def set_base_model(self):
        self.base_model = {
            "type": "MobileNet",
            "params": {
                "input_shape": (224, 224, 3),
                "include_top": False,
                "weights": "imagenet",
            },
        }

    # is called on model build
    def get_base_model(self):
        base_model_params = self.base_model["params"]
        base_model = MobileNet(**base_model_params)
        return base_model
