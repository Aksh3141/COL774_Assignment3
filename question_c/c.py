import sys
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decision_tree import SimpleDecisionTree

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
        tree = SimpleDecisionTree(max_depth=depth)
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
