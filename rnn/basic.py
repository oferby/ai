import collections as coll
from random import shuffle
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = [i for i in range(10)]
shuffle(x)
d = coll.deque()
for i in x:
    d.append(i)
print(d)

W1 = np.random.normal(size=(10, 10))
W2 = np.random.normal(size=(10, 10))
Wh = np.random.normal(size=(10, 10))
h = np.zeros(10)


def predict(x):
    h1 = np.tanh(np.dot(Wh, h) + np.dot(W1, x))
    y = np.dot(W2, h1)
    # print('h:', np.argmax(h))
    y = softmax(y)
    y_max = np.argmax(y)
    return y_max, y[y_max]


for i in range(30):
    x = np.zeros(10)
    x[d[0]] = 1
    print('x:', d[0])

    y_max, y_ = predict(x)
    print('y:', y_max, y_)

    d.rotate(-1)
