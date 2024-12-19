import numpy as np


# noinspection PyPep8Naming,SpellCheckingInspection
class DecisionTreeClassifier:

    def __init__(self, max_depth):
        self.max_depth = max_depth

    # noinspection PyMethodMayBeStatic
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / np.sum(counts)
        return -np.sum(p * np.log2(p))

    def information_gain(self, X, y, feature_idx, threshold):
        # calculate the information gain achieved by splitting on a particular feature and threshold
        entropy_parent = self.entropy(y)

        left_idxs = X[:, feature_idx] < threshold
        left_y = y[left_idxs]
        entropy_left = self.entropy(left_y)
        left_weight = len(left_y) / len(y)

        right_idxs = X[:, feature_idx] >= threshold
        right_y = y[right_idxs]
        entropy_right = self.entropy(right_y)
        right_weight = len(right_y) / len(y)

        entropy_children = left_weight * entropy_left + right_weight * entropy_right
        return entropy_parent - entropy_children

    def split_node(self, X, y, depth):
        n_samples, n_features = X.shape
        best_feature_idx = None
        best_threshold = None
        best_information_gain = -np.inf

        if depth >= self.max_depth:
            count = np.bincount(y)
            return np.argmax(count)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                information_gain = self.information_gain(X, y, feature_idx, threshold)

                if information_gain > best_information_gain:
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_information_gain = information_gain

        if best_information_gain == 0:
            count = np.bincount(y)
            return np.argmax(count)

        left_idxs = X[:, best_feature_idx] < best_threshold
        right_idxs = X[:, best_feature_idx] >= best_threshold

        left_node = self.split_node(X[left_idxs], y[left_idxs], depth + 1)
        right_node = self.split_node(X[right_idxs], y[right_idxs], depth + 1)

        return {
            'feature_idx': best_feature_idx,
            'threshold': best_threshold,
            'left': left_node,
            'right': right_node
        }

    def fit(self, X, y):
        # noinspection PyAttributeOutsideInit
        self._start = self.split_node(X, y, 0)

    def predict_one(self, x):
        node = self._start
        while isinstance(node, dict):
            if x[node['feature_idx']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']

        return node

    def predict(self, X):
        return np.apply_along_axis(self.predict_one, 1, X)
