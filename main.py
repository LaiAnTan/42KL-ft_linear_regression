
import os
import pandas as pd

from srcs.train import train
from srcs.plot import GraphPlotter
from srcs.predict import PredictPrice
from srcs.stats import leastSquares, RSquared
from srcs.helpers import parse, loadCSV


def main():

    parser = parse()

    # load csv
    df = loadCSV("./assets/data.csv")

    data = df.to_numpy()

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()

    elif args.command == 'train':
        print(f"Training with learning rate {args.learning_rate} and max "
              f"steps {args.max_steps}")
        train(data, "./assets/var.txt", learning_rate=args.learning_rate,
              max_steps=args.max_steps)

    elif os.path.exists('./assets/var.txt'):

        match args.command:

            case 'predict':
                PredictPrice("./assets/var.txt").predict()

            case 'plot':
                m_ls, c_ls = leastSquares(data) if args.least_squares else \
                    None, None
                GraphPlotter("./assets/var.txt").plotGraphs(data, m_ls, c_ls)

            case 'rsquared':
                print("R^2 for linear regression via gradient descent: "
                      f"{RSquared('./assets/var.txt').rSquared(data)}")

    else:
        print("Error: You must run the 'train' command before running any "
              "other commands.")


if __name__ == "__main__":
    main()
