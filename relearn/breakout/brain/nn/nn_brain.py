import tensorflow as tf
import time
import numpy as np


def create_nn():
    x = tf.placeholder(tf.float32, [None, 360000], name='inputs')
    W1 = tf.Variable(tf.random_uniform((360000, 1024), -1, 1), name='w1')
    b1 = tf.Variable(tf.zeros((1024,)), name='bios')
    W2 = tf.Variable(tf.random_uniform((1024, 3), -1, 1), name='w2')
    b2 = tf.Variable(tf.zeros((3,)), name='bios')
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1), name='layer1')
    h2 = tf.nn.sigmoid(tf.add(tf.matmul(h1, W2), b2), name='layer2')
    prediction = tf.nn.softmax(h2, name='prediction')
    decision = tf.argmax(prediction, 1)

    label = tf.placeholder(tf.float32, [None, 3], name='label')
    xentropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(xentropy)
    return x, prediction, decision, train_step, label, xentropy


class SimpleNeuralNetworkBrain:
    def __init__(self):
        self.inputs, self.prediction, self.decision, self.train_step, self.label, self.loss = create_nn()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.prev_state = np.zeros((1, 360000))

        self.states = []
        self.values = []
        self.actions = []

        self.gama = 0.9
        self.step = 0

    def choose_action(self, state):
        s_ = np.array(state.reshape(1, 360000))
        state_ = s_ - self.prev_state
        self.prev_state = s_

        self.step += 1
        coin = np.random.randint(3000 + self.step)
        if coin > 2500:
            decision_, prediction_ = self.session.run([self.decision, self.prediction],
                                                      {self.inputs: state_})
        else:
            decision_, prediction_ = self.do_random()
        self.states.append(state_[0])
        self.values.append(prediction_[0])
        self.actions.append(decision_[0])
        decision_ = decision_[0] - 1
        print ('prediction: %s, decision: %s' % (prediction_, decision_))
        return decision_

    def do_random(self):
        p_ = [0.001, 0.001, 0.001]
        a_ = np.random.randint(0, 3)
        p_[a_] = 1
        return [a_], [p_]

    def new_score(self, score):
        self.calc_q_values(score)
        self.train()

    def calc_q_values(self, score):
        print ('calculating Q values')
        decay = self.gama
        for i in range(len(self.actions) - 1, 0, -1):
            self.values[i][self.actions[i]] = score * decay + 0.001
            decay *= self.gama

    def train(self):
        states_ = np.asanyarray(self.states[1:])
        labels_ = np.asanyarray(self.values[1:])
        print ('training network')
        for i in range(10):
            _, loss = self.session.run([self.train_step, self.loss], {self.inputs: states_, self.label: labels_})
            print ('loss: %s', loss)
        self.reset()
        print('done training')

    def start(self):
        print('start new game')
        self.reset()

    def end(self):
        print('end of game')
        self.reset()

    def predict(self, state):
        pass

    def reset(self):
        if len(self.states) > 200:
            self.states = []
            self.values = []
            self.actions = []
