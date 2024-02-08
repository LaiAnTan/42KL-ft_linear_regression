import numpy as np


class StandardScaler():

    """
    Simple reimplementation of scikit-learn's preprocessing.StandardScaler,
    which standardizes data using their mean and standard deviation.
    """

    def __init__(self, data) -> None:

        self.data = data
        self.mean = None
        self.scale = None

    def normalize(self):

        self.mean = [np.mean(self.data[:, 0]), np.mean(self.data[:, 1])]
        self.scale = [np.std(self.data[:, 0]), np.std(self.data[:, 1])]

        scaled = np.ndarray(self.data.shape, np.float32)

        # normalize x
        scaled[:, 0] = (self.data[:, 0] - self.mean[0]) / self.scale[0]

        # normalize y
        scaled[:, 1] = (self.data[:, 1] - self.mean[1]) / self.scale[1]

        return scaled

    def revertCoefficients(self, norm_gradient, norm_intercept):

        # hand calculated algebra
        gradient = norm_gradient * self.scale[1] / self.scale[0]
        intercept = self.mean[1] - (norm_intercept * self.scale[1]) - \
            (self.scale[1] * norm_gradient * self.mean[0] / self.scale[0])

        return gradient, intercept
