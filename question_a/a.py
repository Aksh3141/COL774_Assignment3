import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decision_tree import SimpleDecisionTree

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python question_a/a.py <train_data_path> <val_data_path> <test_data_path> <output_csv_path>")
        sys.exit(1)

    train_path, val_path, test_path, output_csv = sys.argv[1:5]
    if not output_csv.endswith(".csv"):
        output_csv = os.path.join(output_csv, "output.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # ---------------- Load Datasets ----------------
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # ---------------- Split Features & Labels ----------------
    X_train, y_train = train_df.drop(columns=['result']), train_df['result']
    X_val, y_val = val_df.drop(columns=['result']), val_df['result']
    X_test, y_test = test_df.drop(columns=['result']), test_df['result']

    depths = [3,5,10,15,20,25]
    train_accs, val_accs, test_accs = [], [], []

    best_val_acc = -1
    best_tree = None
    best_depth = None
    best_test_pred = None

    print("\n==== Decision Tree Training (No One-Hot Encoding) ====\n")

    # ---------------- Train and Evaluate ----------------
    for depth in depths:
        print(f"Training tree with max_depth={depth}...")
        tree = SimpleDecisionTree(max_depth=depth)
        tree.fit(X_train, y_train)

        y_train_pred = tree.predict(X_train)
        y_val_pred = tree.predict(X_val)
        y_test_pred = tree.predict(X_test)

        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)
        test_acc = np.mean(y_test_pred == y_test)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"Depth={depth:2d} | Train={train_acc:.3f} | Val={val_acc:.3f} | Test={test_acc:.3f}")

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tree = tree
            best_depth = depth
            best_test_pred = y_test_pred

    print(f"\nBest depth selected: {best_depth} (Validation Accuracy = {best_val_acc:.3f})")


    # ---------------- Plot Accuracies ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_accs, marker='o', label='Train')
    plt.plot(depths, val_accs, marker='s', label='Validation')
    plt.plot(depths, test_accs, marker='^', label='Test')
    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs Depth (No One-Hot Encoding)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(output_csv), "post_prune_all_depths.png")
    plt.savefig(plot_path)
    plt.close()

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
        pass
