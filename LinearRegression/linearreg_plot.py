from linearreg import LinearRegression
from linearreg_lib import *

class LinearRegressionPlot:
    def __init__(self, linear: LinearRegression):
        self.linear = linear

    def plot(self):
        a, b = self.linear.a, self.linear.b
        # Generate x-values
        x_values = np.linspace(-10, 10, 100)

        # Calculate y-values
        y_values = a[0] * x_values + b[0]

        # Plot the line
        plt.scatter(self.linear.X_train, self.linear.y_train, color='blue', marker='o', label='Data points')
        plt.plot(x_values, y_values, label=f"y = {a:.2f}x + {b:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Line: y = ax + b")
        plt.grid(True)
        plt.legend()
        plt.show()

