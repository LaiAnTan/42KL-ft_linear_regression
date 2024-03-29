
class PredictPrice:

    def __init__(self, path=None, intercept_name="theta0",
                 slope_name="theta1") -> None:
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

    def estimatePrice(self, milage: int) -> None:
        return self.intercept + (self.slope * milage)

    def predict(self) -> None:

        try:
            milage = float(input('Enter milage (km): '))
            assert milage >= 0
            print(f"Predicted price is {self.estimatePrice(milage)}")
        except Exception:
            return print("Error: milage must be a positive number")


if __name__ == "__main__":
    PredictPrice().predict()
