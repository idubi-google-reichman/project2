from entities.ResNetModel import ResNetModel
from entities.ResNet18Model import ResNet18Model
from entities.MobileNetModel import MobileNetModel
from entities.Profile import Profile
from entities.ENeuralNetworkError import ENeuralNetworkArchitectureError


class ModelFactory:
    @staticmethod
    def create_model(profile_data, train_image_shape_for_resize=None):
        profile_data.update({"TRAIN_IMAGES_SHAPE": train_image_shape_for_resize})

        model_name = profile_data.get("MODEL_NAME")
        sub_model_name = profile_data.get("SUB_MODEL_NAME")

        if model_name == "Resnet":
            # in resnet, shape is required , sine we derive the image that we process from it
            if sub_model_name == "18":
                return ResNet18Model(profile_data)
            else:
                # return ResNetModel(profile_data)
                raise ENeuralNetworkArchitectureError(
                    f" : {model_name},{sub_model_name} "
                )
        elif model_name == "MobileNet":
            #  in mobilenet the shape is 224*224*3 - weset it in the class
            return MobileNetModel(profile_data)
        else:
            raise ENeuralNetworkArchitectureError(f" : {model_name},{sub_model_name} ")
