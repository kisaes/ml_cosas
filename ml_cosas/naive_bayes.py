import numpy as np


# noinspection PyPep8Naming
class NaiveBayesClassifier:

    def __init__(self):
        self.mean = {}
        self.variance = {}
        self.class_probabilities = {}

    def fit(self, X, y):
        # noinspection PyAttributeOutsideInit
        self.classes = np.unique(y)

        for i, c in enumerate(self.classes):
            self.mean[i] = np.mean(X[(y == c)], axis=0)
            self.variance[i] = np.var(X[(y == c)], axis=0)
            self.class_probabilities[i] = np.sum(y == c) / y.shape[0]

    # noinspection PyMethodMayBeStatic
    def _log_likelihood(self, X, mean, variance):
        likelihood = np.exp(-(X - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
        likelihood = np.clip(likelihood, a_min=1e-7, a_max=None)
        return np.log(likelihood)

    def predict(self, X):
        prob = np.zeros((X.shape[0], len(self.classes)))

        for i, _ in enumerate(self.classes):
            log_likelihoods = self._log_likelihood(X, self.mean[i], self.variance[i])
            prob[:, i] = np.sum(log_likelihoods, axis=1) + np.log(self.class_probabilities[i])

        return np.argmax(prob, axis=1)
