'''

    This version uses LSTM from TF
'''

import tensorflow as tf
import collections as coll
from random import shuffle
import numpy as np

x = []
for j in range(3):
    for i in range(10):
        x.append(i)

shuffle(x)
d = coll.deque()
for i in x:
    d.append(i)
print(d)

x_ = [[], [], []]
l_ = []
for i in range(30):
    x1_ = np.zeros(10)
    x1_[d[0]] = 1
    x_[0].append(x1_)
    x2_ = np.zeros(10)
    x2_[d[1]] = 1
    x_[1].append(x1_)
    x3_ = np.zeros(10)
    x3_[d[2]] = 1
    x_[2].append(x1_)
    l1_ = np.zeros(10)
    l1_[d[3]] = 1
    l_.append(l1_)
    d.rotate(-1)


def build_graph():
    # First RNN cell
    x1 = tf.placeholder(tf.float32, [None, 10], name='inputs1')
    W11 = tf.Variable(tf.random_normal((10, 30), -1, 1), name='w11')
    h1 = tf.Variable(tf.zeros((1, 10)))

    W21 = tf.Variable(tf.random_normal((10, 30), -1, 1), name='w21')

    s11 = tf.matmul(x1, W11)
    s21 = tf.matmul(h1, W21)
    s121 = tf.add(s11, s21)

    W31 = tf.Variable(tf.random_normal((30, 10), -1, 1), name='w31')
    h1 = tf.matmul(s121, W31)

    prediction1 = tf.nn.softmax(h1)

    # Second RNN cell

    x2 = tf.placeholder(tf.float32, [None, 10], name='inputs2')
    W12 = tf.Variable(tf.random_normal((10, 30), -1, 1), name='w12')

    W22 = tf.Variable(tf.random_normal((10, 30), -1, 1), name='w22')

    s12 = tf.matmul(x2, W12)
    s22 = tf.matmul(prediction1, W22)
    s122 = tf.add(s12, s22)

    W32 = tf.Variable(tf.random_normal((30, 10), -1, 1), name='w32')
    h2 = tf.matmul(s122, W32)

    prediction2 = tf.nn.softmax(h2)

    # Third RNN cell

    x3 = tf.placeholder(tf.float32, [None, 10], name='inputs3')
    W13 = tf.Variable(tf.random_normal((10, 30), -1, 1), name='w13')

    W23 = tf.Variable(tf.random_normal((10, 30), -1, 1), name='w23')

    s13 = tf.matmul(x3, W13)
    s23 = tf.matmul(prediction2, W23)
    s123 = tf.add(s13, s23)

    W3 = tf.Variable(tf.random_normal((30, 10), -1, 1), name='w33')
    h3 = tf.matmul(s123, W3)

    prediction = tf.nn.softmax(h3)
    y_class = tf.argmax(prediction, axis=1)

    label = tf.placeholder(tf.float32, [None, 10], name='label')
    xentropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(xentropy)
    acc, acc_op = tf.metrics.accuracy(tf.argmax(label, 1), y_class)

    return train_step, xentropy, acc, acc_op, label, y_class, x1, x2, x3


train_step, xentropy, acc, acc_op, label, y_class, x1, x2, x3 = build_graph()

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    for i in range(150000):
        _, loss, acc_, _, l1, y1 = session.run([train_step, xentropy, acc, acc_op, label, y_class],
                                       {label: l_, x1: x_[0], x2: x_[1], x3: x_[2]})
        if i % 5000 == 0:
            print(loss, acc_)
    print('training done')

    for i in range(30):
        x1_ = np.zeros(10)
        x1_[d[0]] = 1
        x2_ = np.zeros(10)
        x2_[d[1]] = 1
        x3_ = np.zeros(10)
        x3_[d[2]] = 1
        h_ = session.run(y_class, {x1: [x1_], x2: [x2_], x3: [x3_]})

        if h_ == d[3]:
            r = '*'
        else:
            r = ' '
        print(d[0], d[1], d[2], 'y:', d[3], 'h:', h_, r)
        d.rotate(-1)
