import numpy as np


# noinspection SpellCheckingInspection
class Adagrad:

    def __init__(self, learning_rate=0.001, epsilon=1e-9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = {}

    def update_parameter(self, parameter, gradient):
        param_key = id(parameter)
        self.cache.setdefault(param_key, np.zeros_like(parameter))
        self.cache[param_key] += gradient ** 2
        parameter += -self.learning_rate * gradient / (np.sqrt(self.cache[param_key]) + self.epsilon)
