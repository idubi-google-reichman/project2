class ENeuralNetworkError(Exception):
    """Exception raised for errors in the neural network issue"""

    def __init__(self, message="There is an issue with the Neural Network Error"):
        self.message = message
        super().__init__(self.message)


class ENeuralNetworkProcessError(Exception):
    """Exception raised for errors in the neural network process issues, like try to get one task before prior need to be completed"""

    def __init__(
        self,
        message="Neural Network Process Error - need to figure out which task is incomplete/not started ",
    ):
        self.message = message
        super().__init__(self.message)


class ENeuralNetworkArchitectureError(ENeuralNetworkError):
    """Exception raised for errors in the neural network architecture issue"""

    def __init__(
        self,
        message="Neural Network Error , Architecture not recognized",
        architecture="",
    ):
        self.message = f"message , ({architecture})  "

        super().__init__(self.message)
