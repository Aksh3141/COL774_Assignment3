import os
import sys
import numpy as np
import cv2
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from neural_network import NeuralNet
import matplotlib.pyplot as plt


def load_dataset(folder_path, num_classes=36, img_size=(32, 32)):
    X, y = [], []
    class_names = sorted(os.listdir(folder_path))
    for label, class_folder in enumerate(class_names):
        class_dir = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Read image using OpenCV (BGR by default)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize to 32x32
                img = cv2.resize(img, img_size)
                # Flatten and normalize
                X.append(img.flatten() / 255.0)
                y.append(label)
            except Exception:
                continue

    return np.array(X), np.array(y), class_names


def evaluate_model(model, X, y_true, class_names):
    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    per_class_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    return per_class_df, avg_precision, avg_recall, avg_f1, y_pred


def main(train_path, test_path, output_folder):
    np.random.seed(42)
    os.makedirs(output_folder, exist_ok=True)

    hidden_units_list = [1, 5, 10, 50, 100]
    lr = 0.01
    batch_size = 32
    epochs = 300
    n_classes = 36
    n_features = 32 * 32 * 3
    tolerance = 1e-4
    patience = 10
    print("Loading training data...")
    X_train, y_train, class_names = load_dataset(train_path)
    print("Training samples:", X_train.shape[0])
    print("Loading test data...")
    X_test, y_test, _ = load_dataset(test_path)
    print("Test samples:", X_test.shape[0])

    summary_records = []
    all_predictions = []  # store predictions vertically

    for hidden_units in hidden_units_list:
        print("\n---------------------------------------------")
        print(f"Training model with {hidden_units} hidden units")
        print("---------------------------------------------")

        model = NeuralNet(
            hidden_layers=[hidden_units],
            input_dim=n_features,
            n_classes=n_classes,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            verbose=True,
            seed=42
        )

        model.fit(X_train, y_train)

        # --- Train metrics ---
        train_metrics, train_prec, train_rec, train_f1, _ = evaluate_model(model, X_train, y_train, class_names)
        # --- Test metrics ---
        test_metrics, test_prec, test_rec, test_f1, y_pred_test = evaluate_model(model, X_test, y_test, class_names)

        # Append predictions vertically
        all_predictions.extend(y_pred_test+1)

        for cls_idx, cls_name in enumerate(class_names):
            summary_records.append({
                "Hidden_Units": hidden_units,
                "Class": cls_name,
                "Train_Precision": train_metrics.loc[cls_idx, "Precision"],
                "Train_Recall": train_metrics.loc[cls_idx, "Recall"],
                "Train_F1": train_metrics.loc[cls_idx, "F1"],
                "Test_Precision": test_metrics.loc[cls_idx, "Precision"],
                "Test_Recall": test_metrics.loc[cls_idx, "Recall"],
                "Test_F1": test_metrics.loc[cls_idx, "F1"]
            })

        print(f"Hidden units = {hidden_units}")
        print(f"  Train: Precision={train_prec:.3f}, Recall={train_rec:.3f}, F1={train_f1:.3f}")
        print(f"  Test : Precision={test_prec:.3f}, Recall={test_rec:.3f}, F1={test_f1:.3f}")

    # --- Save all predictions in a single-column CSV ---
    pred_df = pd.DataFrame(all_predictions, columns=["Prediction"])
    pred_path = os.path.join(output_folder, "prediction_b.csv")
    pred_df.to_csv(pred_path, index=False)

    # --- Save summary metrics ---
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(output_folder, "summary_b.csv"), index=False)
    # Plot F1 vs hidden units
    plt.figure(figsize=(7, 5))
    plt.plot(summary_df["Hidden_Units"], summary_df["Train_F1"], "o--", label="Train F1")
    plt.plot(summary_df["Hidden_Units"], summary_df["Test_F1"], "s-", label="Test F1")
    plt.xlabel("Hidden Units")
    plt.ylabel("Average F1 Score")
    plt.title("Average F1 Score vs Hidden Units")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "f1_vs_hidden_units.png"))
    plt.close()

    print("\nAll results, metrics, and plots saved in:", output_folder)



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python b.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    train_path, test_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    main(train_path, test_path, output_folder)
