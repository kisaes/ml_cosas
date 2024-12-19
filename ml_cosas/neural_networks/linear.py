import numpy as np


def truncated_normal(mean, stddev, size, random):
    result = np.empty(0)
    prod = np.prod(size)

    while result.shape[0] < prod:
        batch = random.normal(mean, stddev, prod)
        valid = batch[(batch >= mean - 2 * stddev) & (batch <= mean + 2 * stddev)]
        result = np.append(result, valid)

    return result[:prod].reshape(size)


# noinspection SpellCheckingInspection
def glorot_normal(size, random: np.random.Generator):
    stddev = 1.0 / max(1.0, np.sum(size) / 2.0)
    stddev = np.sqrt(stddev)
    return truncated_normal(0, stddev, size, random)


class Linear:

    def __init__(self, input_dimensions, output_dimensions, random=np.random.default_rng()):
        self.weights = glorot_normal((input_dimensions, output_dimensions), random)
        self.bias = np.zeros(output_dimensions)

    def __call__(self, inputs):
        # noinspection PyAttributeOutsideInit
        self.inputs = inputs
        return np.matmul(inputs, self.weights) + self.bias

    # noinspection SpellCheckingInspection
    def backprop(self, grads):
        # noinspection PyAttributeOutsideInit
        self._gradients = np.matmul(self.inputs.T, grads), np.sum(grads, axis=0)
        return np.matmul(grads, self.weights.T)

    @property
    def trainable_vars(self):
        return self.weights, self.bias

    @property
    def gradients(self):
        return self._gradients
