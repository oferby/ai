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

    def choose_action(self, state):
        if self.trained:
            action, scores = self.predict(state)
            self.states.append(state)
            self.actions.append(scores)
            if action == 0:
                return -1
            return 1
        else:
            self.states.append(state)
            self.actions.append([0, 1, 0])
            return 0

    def new_score(self, score):
        print('new score %s' % score)

    def start(self):
        print('start new game')

    def end(self):
        print('end of game')

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
