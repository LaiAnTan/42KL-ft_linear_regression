import argparse
import pandas as pd

def loadCSV(path):

    try:
        assert path.split(".")[-1] == 'csv'
        dataframe = pd.read_csv(path)
    except Exception:
        raise AssertionError("Bad file")

    return dataframe

def parse():

    parser = argparse.ArgumentParser(prog="ft_linear_regression",
                                     description="Command line interface for "
                                     "ft_linear_regression")

    subparsers = parser.add_subparsers(dest='command',
                                       help="specify the command to be run")

    train_parser = subparsers.add_parser('train',
                                         help="train the model using gradient "
                                         "descent algorithm. must be run "
                                         "before other commands")
    train_parser.add_argument('-lr', '--learning_rate', type=float,
                              default=0.1,
                              help="specify the learning rate for gradient "
                              "descent. Extreme values will break the "
                              "algorithm.")
    train_parser.add_argument('-ms', '--max_steps', type=int, default=100,
                              help="specify the max steps for gradient "
                              "descent. A large value will make the algorithm "
                              "run for longer. Non integer values or negative "
                              "values will break the algorithm.")
    train_parser.add_argument('-d', '--debug', action='store_const',
                              const=True,
                              help="specify debug mode for gradient "
                              "descent. Enabling debug mode will print info "
                              "to the console.")

    predict_parser = subparsers.add_parser('predict',
                                           help="predict the price of a car "
                                           "based on its milage. uses "
                                           "the trained linear regression.")

    plot_parser = subparsers.add_parser('plot',
                                        help="plot various graphs using data "
                                        "gathered from training.")
    plot_parser.add_argument('-ls', '--least_squares', action='store_true',
                             help="enable the least_squares graph. "
                             "If the dataset is large, it may take a long "
                             "time to load.")

    rsquared_parser = subparsers.add_parser('rsquared',
                                            help="displays the coefficient of "
                                            "determination, R squared, as a "
                                            "measure of precision of the "
                                            "trained linear regression.")

    return parser
