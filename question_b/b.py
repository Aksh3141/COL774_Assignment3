import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decision_tree import SimpleDecisionTree

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python question_b/b.py <train_data_path> <val_data_path> <test_data_path> <output_file_path>")
        sys.exit(1)

    train_path, val_path, test_path, output_csv = sys.argv[1:5]
    if not output_csv.endswith(".csv"):
        output_csv = os.path.join(output_csv, "output.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # ---------------- Load datasets ----------------
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # print("\n==== Dataset Column Types (Before Encoding) ====\n")
    # print("Train Data Types:")
    # print(train_df.dtypes)
    # print("\nValidation Data Types:")
    # print(val_df.dtypes)
    # print("\nTest Data Types:")
    # print(test_df.dtypes)
    # print("=" * 60)

    # ---------------- One-Hot Encoding ----------------
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    multi_cat_cols = [col for col in cat_cols if train_df[col].nunique() > 2]

    # if multi_cat_cols:
    #     print("\nCategorical columns with more than 2 unique values (to be one-hot encoded):")
    #     for col in multi_cat_cols:
    #         print(f" - {col}: {train_df[col].nunique()} unique categories -> {list(train_df[col].unique())[:5]}{'...' if train_df[col].nunique() > 5 else ''}")
    # else:
    #     print("\nNo categorical columns with more than 2 unique values found for one-hot encoding.")

    # Apply one-hot encoding
    train_df = pd.get_dummies(train_df, columns=multi_cat_cols)
    val_df = pd.get_dummies(val_df, columns=multi_cat_cols)
    test_df = pd.get_dummies(test_df, columns=multi_cat_cols)

    # Align columns
    all_cols = set(train_df.columns)
    for df_name, df in [('val', val_df), ('test', test_df)]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[train_df.columns]  # reorder columns
        if df_name == 'val':
            val_df = df
        else:
            test_df = df

    # print("\nAfter One-Hot Encoding:")
    # print(f"Total number of features (excluding target): {len(train_df.columns) - 1}")
    # print("Newly created columns from one-hot encoding:")
    # encoded_cols = [col for col in train_df.columns if any(base in col for base in multi_cat_cols)]
    # for col in encoded_cols:
    #     print(f" - {col}")
    # print("=" * 60)

    # ---------------- Features and Labels ----------------
    X_train, y_train = train_df.drop(columns=['result']), train_df['result']
    X_val, y_val = val_df.drop(columns=['result']), val_df['result']
    X_test, y_test = test_df.drop(columns=['result']), test_df['result']

    # ---------------- Train Decision Tree ----------------
    depths = [15, 25, 35, 45]
    train_accs, val_accs, test_accs = [], [], []
    models = {}

    print("\n==== Decision Tree Training (One-Hot Encoded) ====\n")

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
        models[depth] = (tree, y_test_pred)

        print(f"Depth={depth:2d} | Train={train_acc:.3f} | Val={val_acc:.3f} | Test={test_acc:.3f}")

    # ---------------- Save results (text + plot) ----------------
    results_dir = os.path.dirname(output_csv)
    results_path = os.path.join(results_dir, "one_hot_results.txt")
    with open(results_path, "w") as f:
        f.write("Depth\tTrainAcc\tValAcc\tTestAcc\n")
        for d, tr, va, te in zip(depths, train_accs, val_accs, test_accs):
            f.write(f"{d}\t{tr:.4f}\t{va:.4f}\t{te:.4f}\n")

    # Plot accuracies
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_accs, marker='o', label='Train')
    plt.plot(depths, val_accs, marker='s', label='Validation')
    plt.plot(depths, test_accs, marker='^', label='Test')
    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs Depth (One-Hot Encoded)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "one_hot_accuracy_plot.png")
    plt.savefig(plot_path)
    plt.close()

    # ---------------- Save best test predictions ----------------
    best_depth_idx = int(np.argmax(val_accs))
    best_depth = depths[best_depth_idx]
    _, best_test_pred = models[best_depth]

    print(f"\nBest depth by validation accuracy: {best_depth}")
    print(f"Saving test predictions to CSV: {output_csv}")

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
