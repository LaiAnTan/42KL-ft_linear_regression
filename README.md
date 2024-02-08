# 42KL-ft_linear_regression
An introductory project to machine learning, where gradient descent is used to produce a linear regression for the purpose of predicting the price of a car.

![graphs](./assets/graphs.png)

## Usage

`python main.py <command> <--args>`

Available commands:
- `train` : trains the model via gradient descent algorithm.
  - `--learning_rate=0.1` : sets the learning rate for gradient descent. Extreme values will break the algorithm.
  - `--max_steps=100` : sets the max steps for gradient descent. A large value will make the algorithm run for longer. Non integer values or negative values will break the algorithm.

Commands that can only be used after `train` has been run at least once:
- `predict` : predicts the price of a car based on its milage using the linear regression line which is the product of training.
- `plot` : plots various graphs using data gathered from training.
  - `--least_squares` : enables the least_squares graph. If the dataset is large, it may take a long time to load. 
- `rsquared` : shows the coefficient of determination, $R^2$ as a measure of precision.

## dev/notes/

Dependent variable: price
Independent variable: km

https://hackmd.io/G_d-vmH9Sk6nm5d0XXdzpw?both

