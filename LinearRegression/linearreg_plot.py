from linearreg_lib import *

class LinearRegressionPlot:
    def __init__(self, linear):
        self.linear = linear

    def plot(self, X, X_train, y_train, X_test, y_test):
        y_line = self.linear.predict(X)
        cmap = plt.get_cmap('viridis')
        plt.figure(figsize=(8,6))
        plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
        plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
        plt.plot(X, y_line, color='black', linewidth=2, label='Predictions')
        plt.show()
        plt.legend()
        plt.show()

