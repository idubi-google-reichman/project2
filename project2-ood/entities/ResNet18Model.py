from entities.ResNetModel import ResNetModel
from classification_models.tfkeras import Classifiers


class ResNet18Model(ResNetModel):

    def __init__(self, profile_data):
        self.set_base_model()
        super().__init__(profile_data)

    def set_base_model(self):
        self.base_model = {
            "type": "ResNet18",
            "params": {
                "input_shape": (224, 224, 3),
                "include_top": False,
                "weights": "imagenet",
                "classifier_activation": "softmax",
            },
        }

    def get_base_model(self):
        ResNet18, preprocess_input = Classifiers.get("resnet18")
        base_model_params = self.base_model["params"]
        base_model = ResNet18(**base_model_params)
        return base_model

    # is called on model build
