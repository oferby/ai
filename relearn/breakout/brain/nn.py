from __future__ import division

import numpy as np
import tensorflow as tf
import relearn.breakout.cnn as cnn


def create_cnn():
    action_classifier = tf.estimator.Estimator(
        model_fn=cnn.cnn_model_fn, model_dir="/tmp/cnn_breakout_model")

    return action_classifier


class ReinforcementLearningBrain:
    def __init__(self):
        self.states = []
        self.actions = []
        self.classifier = create_cnn()
        self.trained = False
        self.step = 0
        self.gama = 0.97  # discount factor

    def choose_action(self, state):
        self.step += 1
        if self.trained:
            #  act greedy with random
            coin = np.random.randint(5000 + self.step)
            if coin > 2500:
                action, scores = self.predict(state)
                self.states.append(state)
                self.actions.append(scores)
                if action == 0:
                    return -1
                return 1
            else:
                return self.do_random()
        else:
            self.states.append(state)
            return self.do_random()

    def do_random(self):
        _move = np.random.randint(0, 3)
        new_state = [0.001, 0.001, 0.001]
        new_state[_move] = 1
        move = _move - 1
        self.actions.append(new_state)
        return move

    def new_score(self, score):
        print('new score %s' % score)
        print('num of states: %s' % len(self.states))
        self.calc_q_values(score)
        self.train()
        self.trained = True

    def start(self):
        print('start new game')

    def end(self):
        print('end of game')
        self.states = []
        self.actions = []

    def calc_q_values(self, score):
        i = len(self.actions)
        max = np.argmax(self.actions[i - 1])
        self.actions[i - 1][max] = score
        print('new actions: %s' % self.actions[i - 1])
        max_priv = score
        for j in range(i - 2, 0, -1):
            print('action %s: %s' % (j, self.actions[j]))
            max = np.argmax(self.actions[j])
            new_score = score + self.gama * max_priv
            if new_score > 1:
                self.actions[j][max] = 1
            else:
                self.actions[j][max] = score + self.gama * max_priv
            max_priv = self.actions[j][max]
            print('new actions: %s' % self.actions[j])

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def train(self):
        _x = np.array(self.states, dtype=np.float32)
        _y = np.array(self.actions, dtype=np.float32)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": _x},
            y=_y,
            batch_size=10,
            num_epochs=None,
            shuffle=True)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=10)
        self.trained = True

        self.states = []
        self.actions = []

    def predict(self, state):
        cnn_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": state}, shuffle=False)
        result = self.classifier.predict(cnn_input_fn)
        r = result.next()
        return r['class'], r['probabilities']
