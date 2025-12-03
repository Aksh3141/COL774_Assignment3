import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python question_sklearn_entropy.py <train_data_path> <val_data_path> <test_data_path> <output_csv_path>")
        sys.exit(1)

    train_path, val_path, test_path, output_csv = sys.argv[1:5]
    if not output_csv.endswith(".csv"):
        output_csv = os.path.join(output_csv, "output.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # ---------------- Load Datasets ----------------
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # ---------------- Encode categorical columns ----------------
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    label_encoders = {}

    for col in all_df.columns:
        if all_df[col].dtype == "object":  # If column is string/categorical
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col].astype(str))
            label_encoders[col] = le

    # Split back
    train_df = all_df.iloc[:len(train_df)]
    val_df = all_df.iloc[len(train_df):len(train_df)+len(val_df)]
    test_df = all_df.iloc[len(train_df)+len(val_df):]

    # ---------------- Split Features & Labels ----------------
    X_train, y_train = train_df.drop(columns=['result']), train_df['result']
    X_val, y_val = val_df.drop(columns=['result']), val_df['result']
    X_test, y_test = test_df.drop(columns=['result']), test_df['result']

    # ==========================================================
    # (i) Vary max_depth
    # ==========================================================
    depths = [15, 25, 35, 45]
    train_accs, val_accs, test_accs = [], [], []

    print("\n==== Sklearn Decision Tree (criterion='entropy') ====\n")
    print("Part (i): Varying max_depth...\n")

    best_val_acc = -1
    best_depth = None
    best_model_depth = None
    best_test_pred_depth = None

    for depth in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"Depth={depth:2d} | Train={train_acc:.3f} | Val={val_acc:.3f} | Test={test_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_depth = depth
            best_model_depth = clf
            best_test_pred_depth = y_test_pred

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_accs, marker='o', label='Train')
    plt.plot(depths, val_accs, marker='s', label='Validation')
    plt.plot(depths, test_accs, marker='^', label='Test')
    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs Max Depth (criterion='entropy')")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(output_csv), "sklearn_depth_entropy.png"))
    plt.close()

    print(f"\nBest depth from validation: {best_depth} (Validation Acc = {best_val_acc:.3f})")

    # ==========================================================
    # (ii) Vary ccp_alpha
    # ==========================================================
    print("\nPart (ii): Varying ccp_alpha...\n")

    ccp_alphas = [0.0, 0.0001, 0.0003, 0.0005]
    train_accs_alpha, val_accs_alpha, test_accs_alpha = [], [], []

    best_val_acc_alpha = -1
    best_alpha = None
    best_model_alpha = None
    best_test_pred_alpha = None

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_accs_alpha.append(train_acc)
        val_accs_alpha.append(val_acc)
        test_accs_alpha.append(test_acc)

        print(f"ccp_alpha={alpha:.4f} | Train={train_acc:.3f} | Val={val_acc:.3f} | Test={test_acc:.3f}")

        if val_acc > best_val_acc_alpha:
            best_val_acc_alpha = val_acc
            best_alpha = alpha
            best_model_alpha = clf
            best_test_pred_alpha = y_test_pred

    plt.figure(figsize=(8, 5))
    plt.plot(ccp_alphas, train_accs_alpha, marker='o', label='Train')
    plt.plot(ccp_alphas, val_accs_alpha, marker='s', label='Validation')
    plt.plot(ccp_alphas, test_accs_alpha, marker='^', label='Test')
    plt.xlabel("ccp_alpha")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs ccp_alpha (criterion='entropy')")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(output_csv), "sklearn_ccp_entropy.png"))
    plt.close()

    print(f"\nBest ccp_alpha from validation: {best_alpha} (Validation Acc = {best_val_acc_alpha:.3f})")

    # ==========================================================
    # Save best model predictions
    # ==========================================================
    if best_test_pred_alpha is not None:
        pd.DataFrame({'result': best_test_pred_alpha}).to_csv(output_csv, index=False)
        print(f"\nBest model predictions (ccp_alpha={best_alpha}) saved to {output_csv}")
    else:
        print("\nNo best model found.")
