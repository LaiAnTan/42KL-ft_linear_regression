import sys
import pandas as pd

from srcs.train import train
from srcs.plot import GraphPlotter
from srcs.predict import PredictPrice
from srcs.utils import leastSquares


def loadCSV(path):

    try:
        assert path.split(".")[-1] == 'csv'
        dataframe = pd.read_csv(path)
    except Exception:
        raise AssertionError("Bad file")

    return dataframe


def main():

    df = loadCSV("./assets/data.csv")

    data = df.to_numpy()

    if len(sys.argv) >= 2:
        if sys.argv[1] == "predict":
            prd = PredictPrice("./assets/var.txt")
            try:
                print("Predicted price is "
                      f"{prd.estimatePrice(float(input('Enter milage: ')))}")
            except ValueError:
                return print("Error: milage must be a number")

        elif sys.argv[1] == "train":

            lr = None
            ms = None

            for arg in sys.argv[1:]:
                if arg.startswith("--"):
                    if arg[2:].startswith("learning_rate="):
                        try:
                            lr = float(arg.split("=")[1])
                        except ValueError:
                            return print("Error: "
                                         "--learning_rate only accepts float")
                    if arg[2:].startswith("max_steps="):
                        try:
                            ms = int(arg.split("=")[1])
                        except ValueError:
                            return print("Error: "
                                         "--max_steps only accepts int")

            train(data, "./assets/var.txt", learning_rate=lr, max_steps=ms)

        elif sys.argv[1] == "plot":

            m_ls = None
            c_ls = None

            for arg in sys.argv[1:]:

                if arg == "--least_squares":
                    m_ls, c_ls = leastSquares(data)

            plt = GraphPlotter("./assets/var.txt")
            plt.plotGraphs(data, m_ls, c_ls)



if __name__ == "__main__":
    main()
