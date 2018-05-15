import tensorflow as tf
from tensorflow.contrib import rnn
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


# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3
vocab_size = 10

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


def RNN(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)

    for i in range(5000):
        symbols_in_keys = [[d[0]], [d[1]], [d[2]]]
        symbols_out_onehot = np.zeros(10)
        symbols_out_onehot[d[3]] = 1.0
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: [symbols_in_keys], y: [symbols_out_onehot]})

        if i % 100 == 0:
            print (i, acc, loss)
        d.rotate(-1)

    print('finished training')
    for i in  range(4):
        keys = [[d[0]], [d[1]], [d[2]]]
        onehot_pred = session.run(pred, feed_dict={x: [keys]})
        print(keys, d[3], onehot_pred)
        d.rotate(-3)