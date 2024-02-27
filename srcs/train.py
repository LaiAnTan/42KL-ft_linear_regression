import pandas as pd
import numpy as np

from srcs.helpers import loadCSV
from srcs.scalers import StandardScaler
from srcs.gradientDescent import GradientDescentMSELinear


def writeToFile(path, slope: float, intercept: float,
                cost_history: list[float], slope_history: list[float],
                intercept_history: list[float],
                slope_name="theta1", intercept_name="theta0",
                cost_history_name="cost_history",
                slope_history_name="slope_history",
                intercept_history_name="intercept_history",
                ):

    with open(path, "w") as file:

        file.writelines([f"{intercept_name}={intercept}\n",
                         f"{slope_name}={slope}\n",
                         f"{cost_history_name}="
                         f"{','.join([str(c) for c in cost_history])}\n",
                         f"{slope_history_name}="
                         f"{','.join([str(s) for s in slope_history])}\n",
                         f"{intercept_history_name}="
                         f"{','.join([str(i) for i in intercept_history])}"
                         ])


def train(data: np.ndarray, outfile, learning_rate=0.1, max_steps=100,
          debug=False):

    scaler = StandardScaler(data)

    gdmse = GradientDescentMSELinear(learning_rate=learning_rate,
                                     max_steps=max_steps, debug=debug)

    m, c = gdmse.gradientDescent(scaler.normalize())

    m_gd, c_gd = scaler.revertCoefficients(m, c)

    writeToFile(outfile, m_gd, c_gd, gdmse.cost_history,
                gdmse.slope_history, gdmse.intercept_history)


if __name__ == "__main__":
    data = loadCSV("./assets/data.csv").to_numpy()

    train(data, "./assets/var.txt", debug=True)
