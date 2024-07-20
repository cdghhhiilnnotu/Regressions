import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from logistic import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    acc = np.sum(y_pred == y_true) / len(y_true)
    return acc

regress = LogisticRegression(lr=0.0001, n_iters=1000)
regress.fit(X_train, y_train)
predited = regress.predict(X_test)

print(accuracy(y_test, predited))













