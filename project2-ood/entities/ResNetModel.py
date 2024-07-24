from entities.Profile import Profile


class ResNetModel(Profile):

    def __init__(self, profile_data):
        self.set_base_model()
        super().__init__(profile_data)
