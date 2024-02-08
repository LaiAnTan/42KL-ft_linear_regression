import numpy as np
import matplotlib.pyplot as plt


class GraphPlotter:

    def __init__(self, path=None, slope_name="theta1",
                 intercept_name="theta0",
                 cost_history_name="cost_history",
                 slope_history_name="slope_history",
                 intercept_history_name="intercept_history"):
        self.path = path

        if path is None:
            path = "../assets/var.txt"

        with open(path) as file:

            for line in file:

                tokens = line.split('=')

                if tokens[0] == intercept_name:
                    self.intercept = float(tokens[1])
                elif tokens[0] == slope_name:
                    self.slope = float(tokens[1])
                elif tokens[0] == cost_history_name:
                    self.cost_history = [float(n) for n in
                                         tokens[1].split(',')]
                elif tokens[0] == slope_history_name:
                    self.slope_history = [float(n) for n in
                                          tokens[1].split(',')]
                elif tokens[0] == intercept_history_name:
                    self.intercept_history = [float(n) for n in
                                              tokens[1].split(',')]

    def plotGraphs(self, data: np.ndarray, m_ls=None, c_ls=None):

        fig = plt.figure()

        ax1 = fig.add_subplot(231)

        ax1.scatter(data[:, 0], data[:, 1])
        ax1.set_xlabel("Milage (km)")
        ax1.set_ylabel("Price (Euro)")
        ax1.set_title("Graph of Milage (km) vs Price (Euro)")

        ax2 = fig.add_subplot(232)
        ax2.scatter(data[:, 0], data[:, 1])
        ax2.plot(data[:, 0], (data[:, 0] * self.slope + self.intercept),
                 color="red")
        ax2.set_xlabel("Milage (km)")
        ax2.set_ylabel("Price (Euro)")
        ax2.set_title("Linear Regression via Gradient Descent")

        if m_ls and c_ls:
            ax3 = fig.add_subplot(233)
            ax3.scatter(data[:, 0], data[:, 1])
            ax3.plot(data[:, 0], (data[:, 0] * m_ls + c_ls), color="orange")
            ax3.set_xlabel("Milage (km)")
            ax3.set_ylabel("Price (Euro)")
            ax3.set_title("Linear Regression via Least Squares")

        ax4 = fig.add_subplot(234)
        ax4.plot(list(range(len(self.cost_history))), self.cost_history,
                 color="green")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Cost")
        ax4.set_title("Change in Cost (normalized) over Steps", y=-0.25)

        ax5 = fig.add_subplot(235)
        ax5.plot(list(range(len(self.slope_history))), self.slope_history,
                 color="pink")
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Slope")
        ax5.set_title("Change in Slope value (normalized) over Steps", y=-0.25)

        ax5 = fig.add_subplot(236)
        ax5.plot(list(range(len(self.intercept_history))),
                 self.intercept_history, color="purple")
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Intercept")
        ax5.set_title("Change in Intercept value (normalized) over Steps",
                      y=-0.25)

        plt.show()
