from linearreg_lib import *
from linearreg_plot import *
from linearreg_plot import LinearRegressionPlot

class LinearRegression:

    def __init__(self, train, lr=0.001):
        self.train = train
        self.num_sample, self.num_feat = train.shape
        self.X_train = train[:,0]
        self.y_train = train[:,1]
        self.alpha = lr

    def random_line(self):
        self.a = np.random.randn()
        self.b = np.random.randn()

    def pred_y(self):
        return self.a * self.X_train + self.b

    def distance(self):
        return np.abs(self.a * self.X_train - self.y_train + self.b) / np.sqrt(self.a**2 + 1)

    def training(self):
        self.random_line()
        
        for i in range(3):
            self.dw = (1/self.num_sample)*2*self.X_train*(self.pred_y() - self.y_train)
            self.db = (1/self.num_sample)*2*(self.pred_y() - self.y_train)

            self.a = self.a - self.alpha*self.dw
            self.b = self.b - self.alpha*self.db

            self.loss = (1/self.num_sample)*np.sum(self.y_train - (self.a*self.X_train + self.b))**2

if __name__ == "__main__":

    train, _ = make_blobs(n_samples=10, n_features=2)

    linear = LinearRegression(train)

    linear.training()

    # linear_plot = LinearRegressionPlot(linear)
    # linear_plot.plot()
    
