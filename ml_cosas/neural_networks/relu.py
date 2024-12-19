import numpy as np


class ReLU:

    def __call__(self, inputs):
        # noinspection PyAttributeOutsideInit
        self.inputs = inputs
        return np.maximum(0, inputs)

    # noinspection SpellCheckingInspection
    def backprop(self, grads):
        grads = grads.copy()
        grads[self.inputs <= 0] = 0
        return grads
