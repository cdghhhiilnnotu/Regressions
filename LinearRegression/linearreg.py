from linearreg_lib import *
from linearreg_plot import *

class LinearRegression:

    def __init__(self, lr=0.001, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y  ):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weight) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred
    
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

if __name__ == "__main__":

    X,y = make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    pred = linear.predict(X_test)

    print(linear.loss(y_test, pred))

    linear_plot = LinearRegressionPlot(linear)
    linear_plot.plot(X, X_train, y_train, X_test, y_test)






    
