import numpy as np
import pandas as pd
from collections import Counter
import sys
import os
from copy import deepcopy
import matplotlib.pyplot as plt


class SimpleDecisionTreeGini:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    # ---------------------- Gini Impurity ----------------------
    def _gini(self, y):
        """Compute Gini impurity of label distribution."""
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def _gain(self, parent_y, children_y_list):
        """Gini impurity reduction from splitting parent set into children."""
        base_gini = self._gini(parent_y)
        n = len(parent_y)
        child_gini = 0.0
        for child_y in children_y_list:
            weight = len(child_y) / n
            child_gini += weight * self._gini(child_y)
        return base_gini - child_gini

    # ------------------------------------------------------------
    def _choose_best_attribute(self, X, y):
        """Return attribute (and split rule) giving max Gini gain."""
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

# ---------------- Helper Functions ----------------
def one_hot_encode(df, exclude_cols=[]):
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col in exclude_cols:
            continue
        if df_encoded[col].nunique() > 2:
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
    return df_encoded

def count_nodes(node):
    if 'label' in node:
        return 1
    return 1 + sum(count_nodes(child) for child in node['children'].values())

def collect_labels(node):
    if 'label' in node:
        return [node['label']]
    labels = []
    for child in node['children'].values():
        labels.extend(collect_labels(child))
    return labels

def all_non_leaf_paths(node, path=None):
    """
    Return list of paths (each path is a list of keys) to all non-leaf nodes.
    """
    if path is None:
        path = []
    paths = []
    if 'label' not in node:
        paths.append(list(path))
        for key, child in node['children'].items():
            paths.extend(all_non_leaf_paths(child, path + [key]))
    return paths

def get_parent_and_key(root, path):
    """
    Given a path (list of keys) from root to node, return (parent_node, key, node).
    If path is empty, parent is None, key is None, node is root itself.
    """
    if not path:
        return None, None, root
    parent = root
    for key in path[:-1]:
        parent = parent['children'][key]
    key = path[-1]
    node = parent['children'][key]
    return parent, key, node

def predict_with_root(root, X):
    preds = []
    for i in range(len(X)):
        x = X.iloc[i]
        node = root
        while 'label' not in node:
            attr = node['feature']
            if node.get('numeric', False):
                # numeric split: left / right
                node = node['children']['left'] if x[attr] <= node['threshold'] else node['children']['right']
            else:
                # categorical: try exact match, otherwise fallback to first child
                node = node['children'].get(x[attr], list(node['children'].values())[0])
        preds.append(node['label'])
    return np.array(preds)

def prune_tree(tree, X_val, y_val, X_train, y_train, X_test=None, y_test=None):
    """
    Greedy post-pruning using validation accuracy. Operates on a deep-copied root
    and replaces child pointers on parent nodes by leaf dicts to avoid shared-reference issues.
    Returns: pruned_root, best_node_count, best_train_acc, best_val_acc, best_test_acc
    """
    pruned_root = deepcopy(tree.root)

    # initial best metrics (full tree)
    best_val_acc = np.mean(predict_with_root(pruned_root, X_val) == y_val)
    best_train_acc = np.mean(predict_with_root(pruned_root, X_train) == y_train)
    best_test_acc = np.mean(predict_with_root(pruned_root, X_test) == y_test) if (X_test is not None and y_test is not None) else None
    best_node_count = count_nodes(pruned_root)

    # Repeat until no single-node pruning increases/maintains validation accuracy
    while True:
        non_leaf_paths = all_non_leaf_paths(pruned_root)
        if not non_leaf_paths:
            break

        pruned_this_round = False

        # Try each candidate node (by path). Greedy: accept first that doesn't reduce val acc.
        for path in non_leaf_paths:
            parent, key, node = get_parent_and_key(pruned_root, path)
            backup_subtree = deepcopy(node)   # deep backup

            # Create leaf with majority label of this subtree
            majority_label = Counter(collect_labels(node)).most_common(1)[0][0]
            leaf = {'label': majority_label}

            # Replace the node at parent[key] (or replace root if parent is None)
            if parent is None:
                # replacing root
                tentative_root = leaf
            else:
                # we must not mutate backup structures referenced elsewhere; create a deep copy of root for evaluation
                # but to keep it efficient we will modify pruned_root in-place and revert if needed by restoring backup_subtree
                parent['children'][key] = leaf
                tentative_root = pruned_root

            # Evaluate validation accuracy with the tentative pruning
            new_val_acc = np.mean(predict_with_root(tentative_root, X_val) == y_val)

            if new_val_acc >= best_val_acc:
                # Accept pruning
                pruned_this_round = True
                if parent is None:
                    pruned_root = leaf
                # update best metrics
                best_val_acc = new_val_acc
                best_train_acc = np.mean(predict_with_root(pruned_root, X_train) == y_train)
                if X_test is not None and y_test is not None:
                    best_test_acc = np.mean(predict_with_root(pruned_root, X_test) == y_test)
                best_node_count = count_nodes(pruned_root)
                # break to restart scanning from the new root
                break
            else:
                # Revert change
                if parent is None:
                    pruned_root = backup_subtree
                else:
                    parent['children'][key] = backup_subtree
                # continue trying other nodes

        if not pruned_this_round:
            break

    return pruned_root, best_node_count, best_train_acc, best_val_acc, best_test_acc


# ---------------- Main ----------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python c.py <train_path> <val_path> <test_path> <output_csv_path>")
        sys.exit(1)

    train_path, val_path, test_path, output_csv = sys.argv[1:5]

    # Ensure CSV file ends with .csv
    if not output_csv.endswith(".csv"):
        output_csv = os.path.join(output_csv, "output.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    target_col = 'result'

    # One-Hot Encoding
    X_train = one_hot_encode(train_df.drop(columns=[target_col]))
    y_train = train_df[target_col]

    X_val = one_hot_encode(val_df.drop(columns=[target_col]))
    y_val = val_df[target_col]

    X_test = one_hot_encode(test_df.drop(columns=[target_col]))
    y_test = test_df[target_col]

    # Align columns
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    depths = [15,25,35,45]
    all_stats = {}
    best_test_pred = None

    train_accs, val_accs, test_accs = [], [], []
    nodes = []

    for depth in depths:
        print(f"\nTraining full tree with max_depth={depth}...")
        tree = SimpleDecisionTreeGini(max_depth=depth)
        tree.fit(X_train, y_train)
        y_val_pred = tree.predict(X_val)
        print("Nodes before pruning:", count_nodes(tree.root))
        print("Validation Accuracy before Pruning:", np.mean(y_val_pred == y_val))

        pruned_root, node_count, train_acc, val_acc, test_acc = prune_tree(tree, X_val, y_val, X_train, y_train, X_test, y_test)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        nodes.append(node_count)

        print("--------------------------------")
        print("Depth", depth)
        print("Training Accuracy: ", train_acc)
        print("Validation accuracy: ", val_acc)
        print("Test accuracy: ", test_acc)
        print("Number of nodes: ", node_count)

        all_stats[depth] = {
            'nodes': node_count,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_accs': test_accs,
            'pruned_root': pruned_root
        }

        # If you want best_test_pred for a specific depth, update this accordingly
        if depth == 35:
            best_test_pred = predict_with_root(pruned_root, X_test)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(nodes, train_accs, marker='o', linestyle='-', label=f'Train')
    plt.plot(nodes, val_accs, marker='s', linestyle='-', label=f'Val')
    plt.plot(nodes, test_accs, marker='^', linestyle='-', label=f'Test')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.title("Post-Pruning Accuracy vs Nodes (All Depths)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(output_csv), "post_prune_all_depths.png")
    plt.savefig(plot_path)
    #plt.show()
    plt.close()

    print(f"Combined pruning plot saved as {plot_path}")
    if best_test_pred is not None:
        if not os.path.isfile(output_csv):
            pd.DataFrame({'result': best_test_pred}).to_csv(output_csv, index=False)
            print(f"Best pruned tree test predictions saved to {output_csv}")
        else:
            existing = pd.read_csv(output_csv)
            existing['result'] = best_test_pred
            existing.to_csv(output_csv, index=False)
            print(f"Updated 'result' column in existing file: {output_csv}")
    else:
        print("No best_test_pred was produced (depth 35 not in depths). Output CSV not updated.")
