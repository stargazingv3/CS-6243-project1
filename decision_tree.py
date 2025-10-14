# Zijie Zhang, Sep 2025
# Decision Tree implementation in a modular class structure (Corrected)

import numpy as np
from collections import Counter

class Node:
    """A node in the decision tree."""
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.branches = {}

class DecisionTreeClassifier:
    """A Decision Tree classifier implemented from scratch."""

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.feature_names = None

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities[probabilities > 0]))

    def _information_gain(self, y, splits):
        parent_entropy = self._entropy(y)
        total_samples = len(y)
        if total_samples == 0:
            return 0
        children_entropy = sum(
            (len(split) / total_samples) * self._entropy(y[split]) for split in splits
        )
        return parent_entropy - children_entropy

    def _majority_label(self, y):
        if len(y) == 0:
            return None
        return Counter(y).most_common(1)[0][0]

    # <-- CHANGED: Added split_features parameter
    def _best_split(self, X, y, split_features):
        best_gain, best_feature, best_threshold, best_splits = -1, None, None, None
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_name = self.feature_names[feature_idx]

            # <-- CHANGED: Added check to skip used categorical features
            if feature_name in split_features:
                continue

            col = X[:, feature_idx]
            
            try: # Numerical
                col_float = col.astype(float)
                unique_values = np.unique(col_float)
                if len(unique_values) <= 1:
                    continue
                
                for value in unique_values:
                    left_indices = np.where(col_float <= value)[0]
                    right_indices = np.where(col_float > value)[0]
                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue
                    gain = self._information_gain(y, [left_indices, right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_name
                        best_threshold = value
                        best_splits = [left_indices, right_indices]
            
            except ValueError: # Categorical
                unique_values = np.unique(col)
                if len(unique_values) <= 1:
                    continue
                splits = [np.where(col == val)[0] for val in unique_values]
                gain = self._information_gain(y, splits)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_name
                    best_threshold = None
                    best_splits = splits
        
        return best_feature, best_threshold, best_gain, best_splits

    # <-- CHANGED: Added split_features parameter with default value
    def _grow_tree(self, X, y, depth=0, split_features=None):
        # <-- CHANGED: Initialize list if it's the first call
        if split_features is None:
            split_features = []

        if len(set(y)) == 1 or len(y) == 0 or (self.max_depth and depth >= self.max_depth):
            return Node(is_leaf=True, prediction=self._majority_label(y))

        # <-- CHANGED: Pass split_features to _best_split
        feat, thr, gain, splits = self._best_split(X, y, split_features)
        
        if feat is None or gain <= 1e-12:
            return Node(is_leaf=True, prediction=self._majority_label(y))

        node = Node(feature=feat, threshold=thr)
        
        if thr is not None: # Numerical split
            # For numerical, we don't add to split_features, as you can split on it again
            node.branches["<="] = self._grow_tree(X[splits[0]], y[splits[0]], depth + 1, split_features)
            node.branches[">"] = self._grow_tree(X[splits[1]], y[splits[1]], depth + 1, split_features)
        else: # Categorical split
            feat_idx = self.feature_names.index(feat)
            unique_values = np.unique(X[:, feat_idx])
            # <-- CHANGED: Pass the updated list of used features
            new_split_features = split_features + [feat]
            for val, split_indices in zip(unique_values, splits):
                node.branches[val] = self._grow_tree(X[split_indices], y[split_indices], depth + 1, new_split_features)
        
        return node
        
    def fit(self, X, y, feature_names=None):
        if feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            self.feature_names = list(feature_names)
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.prediction
        feature_idx = self.feature_names.index(node.feature)
        value = x[feature_idx]
        if node.threshold is not None:
            key = "<=" if float(value) <= node.threshold else ">"
        else:
            key = value
        branch = node.branches.get(key)
        if branch is None:
            # This part is tricky; a robust implementation would have a better fallback.
            # For now, this might not even be reached with the fix.
            return self._majority_label([]) 
        return self._traverse_tree(x, branch)
        
    def predict(self, X):
        if self.root is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # (print_tree methods remain the same)
    def _print_recursive(self, node, depth=0):
        if node is None: return
        if node.is_leaf:
            print(' | ' * depth + "+-> " + "Predict:", node.prediction)
            return
        if node.threshold is not None:
            print(' | ' * depth + "+-> " + node.feature + " <= " + str(node.threshold))
            self._print_recursive(node.branches["<="], depth + 1)
            print(' | ' * depth + "+-> " + node.feature + " > " + str(node.threshold))
            self._print_recursive(node.branches[">"], depth + 1)
        else:
            for key, branch in node.branches.items():
                print(' | ' * depth + "+-> " + node.feature + " is " + str(key))
                self._print_recursive(branch, depth + 1)

    def print_tree(self):
        if self.root is None:
            print("Tree has not been fitted yet.")
        else:
            print("root")
            self._print_recursive(self.root)

# This block allows the file to be run standalone for testing
if __name__ == "__main__":
    filepath = "./data/example.csv"
    print(f"--- Running DecisionTreeClassifier Standalone Test ---")
    print(f"Loading data from: {filepath}")

    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
    
    rawdata = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=object)
    y = rawdata[:, -1]
    X = rawdata[:, :-1]
    feature_names = header[:-1]

    # --- Decision Tree Execution ---
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y, feature_names=feature_names)
    
    y_pred = tree.predict(X)
    
    # --- Evaluation ---
    accuracy = np.mean(y_pred == y)
    print(f"\nAccuracy on training data: {accuracy:.4f}\n")
    
    # --- Visualization ---
    print("Tree Structure:")
    tree.print_tree()