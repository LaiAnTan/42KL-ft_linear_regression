import numpy as np


class GradientDescentMSELinear:

    """
    Implementation of gradient descent algorithm with Mean Squared Error (MSE)
    cost function, which optimises parameters for a linear regression line.
    """

    def __init__(self, learning_rate=0.1, max_steps=100,
                 starting_slope=0, starting_intercept=0,
                 debug=False) -> None:

        if learning_rate is None:
            learning_rate = 0.1
        if max_steps is None:
            max_steps = 100

        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.slope = starting_slope
        self.intercept = starting_intercept
        self.debug = debug

        self.cost_history = []
        self.slope_history = []
        self.intercept_history = []

    def estimate(self, x):
        return (self.slope * x) + self.intercept

    def gradientDescent(self, data: np.ndarray):

        self.cost_history = []
        self.slope_history = []
        self.intercept_history = []

        for step in range(1, self.max_steps + 1):

            N = data.shape[0]

            slope_grad = 0  # d Cost / d slope
            intercept_grad = 0  # d Cost / d intercept

            for x, y in data:
                slope_grad += (self.estimate(x) - y) * x
                intercept_grad += (self.estimate(x) - y)

            self.slope -= self.learning_rate * (1 / N) * slope_grad
            self.intercept -= self.learning_rate * (1 / N) * intercept_grad

            cost = self.mean_squared_error(data)

            self.cost_history.append(cost)
            self.slope_history.append(self.slope)
            self.intercept_history.append(self.intercept)

            if self.debug:
                print(f"Step: {step}, Cost: {cost},"
                      f"slope = {self.slope}, intercept = {self.intercept}")

        return self.slope, self.intercept

    def mean_squared_error(self, data):
        return np.sum((data[:, 1] - self.estimate(data[:, 0])) ** 2, axis=0) \
            / data.shape[0]
