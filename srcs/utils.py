import numpy as np


def leastSquares(arr: np.ndarray):

    mean = np.mean(arr, axis=0)

    x_mean_diffs = arr[:, 0] - mean[0]  # mean(x) - x
    y_mean_diffs = arr[:, 1] - mean[1]  # mean(y) - y

    slope = np.sum(x_mean_diffs * y_mean_diffs) / np.sum(x_mean_diffs ** 2)

    intercept = mean[1] - (slope * mean[0])

    return slope, intercept


def rSquared(arr: np.ndarray, function: callable):

    print(arr)

    mean = np.mean(arr, axis=0)

    y_mean_diffs = arr[:, 1] - mean[1]

    predicted_y_values = np.vectorize(function)(arr[:, 0])

    predicted_y_mean_diffs = predicted_y_values - mean[1]

    r_squared = np.sum(predicted_y_mean_diffs ** 2) / np.sum(y_mean_diffs ** 2)

    return r_squared
