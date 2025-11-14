import numpy as np
import pandas as pd
from collections import Counter

class SimpleDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    # ---------------------- Basic Helpers ----------------------
    def _entropy(self, y):
        """Compute entropy of label distribution."""
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _gain(self, parent_y, children_y_list):
        """Information gain from splitting parent set into children."""
        base_entropy = self._entropy(parent_y)
        n = len(parent_y)
        child_entropy = 0.0
        for child_y in children_y_list:
            weight = len(child_y) / n
            child_entropy += weight * self._entropy(child_y)
        return base_entropy - child_entropy

    # ------------------------------------------------------------
    def _choose_best_attribute(self, X, y):
        """Return attribute (and split rule) giving max info gain."""
        best_gain = -float('inf')
        best_feature = None
        best_rule = None
        is_num = False
        median_val = None

        for feature in X.columns:
            col_data = X[feature]

            if np.issubdtype(col_data.dtype, np.number):
                # For numeric features, split by median value
                med = np.median(col_data)
                left_mask = col_data <= med
                right_mask = col_data > med
                split_sets = [y[left_mask], y[right_mask]]
                gain = self._gain(y, split_sets)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    is_num = True
                    median_val = med
                    best_rule = (left_mask, right_mask)

            else:
                # For categorical feature, do k-way split
                uniq_vals = col_data.unique()
                splits = [y[col_data == val] for val in uniq_vals]
                gain = self._gain(y, splits)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    is_num = False
                    best_rule = uniq_vals

        return best_feature, best_gain, best_rule, is_num, median_val

    # ------------------------------------------------------------
    def _grow(self, X, y, depth=0):
        """Recursively construct the decision tree."""
        # Base cases
        if len(set(y)) == 1:
            return {'label': y.iloc[0]}
        if self.max_depth is not None and depth >= self.max_depth:
            return {'label': Counter(y).most_common(1)[0][0]}
        if X.empty:
            return {'label': Counter(y).most_common(1)[0][0]}

        # Choose the best attribute to split
        feat, gain, rule, numeric, threshold = self._choose_best_attribute(X, y)
        if gain <= 0 or feat is None:
            return {'label': Counter(y).most_common(1)[0][0]}

        node = {
            'feature': feat,
            'numeric': numeric,
            'children': {}
        }

        # Split by feature type
        if numeric:
            left_idx = X[feat] <= threshold
            right_idx = X[feat] > threshold
            node['threshold'] = threshold
            node['children']['left'] = self._grow(X[left_idx], y[left_idx], depth + 1)
            node['children']['right'] = self._grow(X[right_idx], y[right_idx], depth + 1)
        else:
            for val in rule:
                mask = X[feat] == val
                node['children'][val] = self._grow(X[mask], y[mask], depth + 1)

        return node

    # ------------------------------------------------------------
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, (list, np.ndarray)):
            y = pd.Series(y)

        self.majority_label = Counter(y).most_common(1)[0][0]  
        self.root = self._grow(X, y)


    # ------------------------------------------------------------
    def _predict_one(self, x, node):
        if node is None or not isinstance(node, dict):
            return self.majority_label

        # Leaf node
        if 'label' in node:
            return node['label']

        # Internal node: use 'feature' and 'numeric' keys (consistent with _grow)
        attr = node['feature']
        if node['numeric']:
            threshold = node['threshold']
            if x[attr] <= threshold:
                return self._predict_one(x, node['children']['left'])
            else:
                return self._predict_one(x, node['children']['right'])
        else:
            val = x[attr]
            if val in node['children']:
                return self._predict_one(x, node['children'][val])
            else:
                # unseen category: fallback to majority of children labels or global
                child_labels = [child.get('label') for child in node['children'].values() if 'label' in child]
                if child_labels:
                    return Counter(child_labels).most_common(1)[0][0]
                else:
                    return self.majority_label




    # ------------------------------------------------------------
    def predict(self, X):
        """Predict for a batch of examples."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        preds = [self._predict_one(X.iloc[i], self.root) for i in range(len(X))]
        return np.array(preds)
