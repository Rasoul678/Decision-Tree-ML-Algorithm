import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initialize the Decision Tree classifier.

        Parameters:
        - max_depth: Maximum depth of the tree (controls complexity)
        - min_samples_split: Minimum number of samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """
        Build the decision tree from the training data.

        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target values (n_samples,)
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters:
        - X: Feature matrix for current node
        - y: Target values for current node
        - depth: Current depth of the node

        Returns:
        - A dictionary representing the node (either decision node or leaf node)
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_labels == 1 or \
                n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return {'value': leaf_value}

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)

        # If no split improves purity, return a leaf
        if best_feature is None:
            return {'value': self._most_common_label(y)}

        # Split the data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs

        # Recursively grow left and right subtrees
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        # Return the decision node
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left,
            'right': right
        }

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        Parameters:
        - X: Feature matrix
        - y: Target values

        Returns:
        - best_feature: Index of best feature to split on
        - best_threshold: Best threshold value for the split
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        current_entropy = self._entropy(y)

        for feature in range(self.n_features):
            # Get all unique values in this feature to consider as thresholds
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # Split the data
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs

                # Skip if split doesn't divide the data
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue

                # Calculate information gain
                gain = self._information_gain(y, left_idxs, right_idxs, current_entropy)

                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, left_idxs, right_idxs, current_entropy):
        """
        Calculate information gain for a split.

        Parameters:
        - y: Target values
        - left_idxs: Indices of left split
        - right_idxs: Indices of right split
        - current_entropy: Entropy before the split

        Returns:
        - information_gain: The reduction in entropy from the split
        """
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])

        if n_left == 0 or n_right == 0:
            return 0

        # Weighted average of child entropies
        child_entropy = (n_left / n) * self._entropy(y[left_idxs]) + \
                        (n_right / n) * self._entropy(y[right_idxs])

        # Information gain is entropy before minus weighted entropy after
        return current_entropy - child_entropy

    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Parameters:
        - y: Array of class labels

        Returns:
        - entropy: The entropy of the label distribution
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Find the most common class label in a set.

        Parameters:
        - y: Array of class labels

        Returns:
        - The most frequent class label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X: Feature matrix (n_samples, n_features)

        Returns:
        - predictions: Array of predicted class labels
        """
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        """
        Recursively traverse the tree to make a prediction for a single sample.

        Parameters:
        - x: Single sample (n_features,)
        - node: Current node in the tree

        Returns:
        - Predicted class label
        """
        if 'value' in node:
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])