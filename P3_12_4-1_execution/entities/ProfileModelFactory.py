import 

class ModelFactory:
    @staticmethod
    def create_model(profile_data):
        model_name = profile_data.get("MODEL_NAME")
        if model_name == "Resnet":
            return ResnetModel(profile_data)
        elif model_name == "MobileNet":
            return MobileNetModel(profile_data)
        else:
            raise
