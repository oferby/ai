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
        gama = 0.97  # discount factor

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
        new_state = [0, 0, 0]
        new_state[_move] = 1
        move = _move - 1
        self.actions.append(new_state)
        return move

    def new_score(self, score):
        print('new score %s' % score)
        print('num of states: %s' % len(self.states))
        self.calc_q_values(score)

    def start(self):
        print('start new game')

    def end(self):
        print('end of game')
        self.states = []
        self.actions = []

    def calc_q_values(self, score):
        i = len(self.actions)

    def train(self):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.states},
            y=self.actions,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=100)
        self.trained = True

    def predict(self, state):
        cnn_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": state}, shuffle=False)
        result = self.classifier.predict(cnn_input_fn)
        r = result.next()
        return r['class'], r['probabilities']
