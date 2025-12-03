import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python e_random_forest_label_encode.py <train_path> <val_path> <test_path> <output_csv_path>")
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

    # ---------------- Label Encode Categorical Features ----------------
    cat_cols = X_train.select_dtypes(include='object').columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    # ---------------- Grid Search ----------------
    param_grid = {
        'n_estimators': [50, 150, 250, 350],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_samples_split': [2, 4, 6, 8, 10]
    }

    best_val_acc = -1
    best_params = None
    best_model = None
    results = []

    total_configs = len(list(ParameterGrid(param_grid)))
    config_count = 0

    for params in ParameterGrid(param_grid):
        config_count += 1
        clf = RandomForestClassifier(
            criterion='entropy',
            n_estimators=params['n_estimators'],
            max_features=params['max_features'],
            min_samples_split=params['min_samples_split'],
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        test_acc = clf.score(X_test, y_test)
        oob_acc = clf.oob_score_

        results.append({
            'params': params,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'oob_acc': oob_acc
        })

        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_model = clf

        # --- Print progress for this configuration ---
        print(f"[{config_count}/{total_configs}] Finished training with params: {params}")
        print(f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, Test Acc={test_acc:.3f}, OOB Acc={oob_acc:.3f}\n")


    # ---------------- Report ----------------
    print("Best Hyperparameters:", best_params)
    print(f"Train Accuracy: {best_model.score(X_train, y_train):.3f}")
    print(f"OOB Accuracy:   {best_model.oob_score_:.3f}")
    print(f"Validation Accuracy: {best_val_acc:.3f}")
    print(f"Test Accuracy:  {best_model.score(X_test, y_test):.3f}")

    # Save best test predictions
    best_test_pred = best_model.predict(X_test)
    pd.DataFrame({'result': best_test_pred}).to_csv(output_csv, index=False)
    print(f"Best model test predictions saved to {output_csv}")
