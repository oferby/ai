import numpy as np


class RandomBrain:
    def choose_action(self, state):
        move = np.random.randint(0, 3) - 1
        return move

    def new_score(self, score):
        pass

    def start(self):
        pass

    def end(self):
        pass
