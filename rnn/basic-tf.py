'''

    First try.
    Less neurons in the hidden layer and the actual output is not fed into the next cell.

'''

import tensorflow as tf
import collections as coll
from random import shuffle
import numpy as np

x = [i for i in range(10)]
shuffle(x)
d = coll.deque()
for i in x:
    d.append(i)
print(d)

x = tf.placeholder(tf.float32, [None, 10], name='inputs')
W1 = tf.Variable(tf.random_uniform((10, 10), -1, 1), name='w1')
h = tf.Variable(tf.zeros((1, 10)))

W2 = tf.Variable(tf.random_uniform((10, 10), -1, 1), name='w2')

s1 = tf.matmul(x, W1)
s2 = tf.matmul(h, W2)
h = tf.add(s1, s2)

W3 = tf.Variable(tf.random_uniform((10, 10), -1, 1), name='w3')
y = tf.matmul(h, W3)
prediction = tf.nn.softmax(y)
y_class = tf.argmax(prediction, axis=1)

label = tf.placeholder(tf.float32, [None, 10], name='label')
xentropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(xentropy)

x_ = []
l_ = []
for i in range(10):
    x1_ = np.zeros(10)
    x1_[d[0]] = 1
    x_.append(x1_)
    l1_ = np.zeros(10)
    l1_[d[1]] = 1
    l_.append(l1_)
    d.rotate(-1)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(500):
        _, loss = session.run([train_step, xentropy], {label: l_, x: x_})
        # print(loss)
    print('training done')

    for i in range(10):
        x_ = np.zeros(10)
        x_[d[0]] = 1
        h_ = session.run(y_class, {x: [x_]})
        print('x:', d[0], 'h:', h_)
        d.rotate(-1)
