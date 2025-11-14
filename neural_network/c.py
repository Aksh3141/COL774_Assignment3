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
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
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

    # Depth configurations (list of hidden layer sizes)
    hidden_layer_configs = [
        [512],
        [512, 256],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]

    lr = 0.01
    batch_size = 32
    epochs = 300
    n_classes = 36
    n_features = 32 * 32 * 3

    print("Loading training data...")
    X_train, y_train, class_names = load_dataset(train_path)
    print("Training samples:", X_train.shape[0])

    print("Loading test data...")
    X_test, y_test, _ = load_dataset(test_path)
    print("Test samples:", X_test.shape[0])

    summary_records = []
    all_predictions = []

    for i, hidden_layers in enumerate(hidden_layer_configs, start=1):
        print("\n---------------------------------------------")
        print(f"Training model with depth={i}, hidden layers={hidden_layers}")
        print("---------------------------------------------")

        model = NeuralNet(
            hidden_layers=hidden_layers,
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

        all_predictions.extend(y_pred_test+1)

        summary_records.append({
            "Depth": i,
            "Hidden_Layers": str(hidden_layers),
            "Train_Precision": train_prec,
            "Train_Recall": train_rec,
            "Train_F1": train_f1,
            "Test_Precision": test_prec,
            "Test_Recall": test_rec,
            "Test_F1": test_f1
        })

        print(f"Depth = {i} ({hidden_layers})")
        print(f"  Train: Precision={train_prec:.3f}, Recall={train_rec:.3f}, F1={train_f1:.3f}")
        print(f"  Test : Precision={test_prec:.3f}, Recall={test_rec:.3f}, F1={test_f1:.3f}")

    # --- Save predictions ---
    pred_df = pd.DataFrame(all_predictions, columns=["Prediction"])
    pred_df.to_csv(os.path.join(output_folder, "prediction_c.csv"), index=False)

    # --- Save summary metrics ---
    summary_df = pd.DataFrame(summary_records)
    # summary_df.to_csv(os.path.join(output_folder, "summary_c.csv"), index=False)

    # --- Plot F1 vs depth ---
    plt.figure(figsize=(7, 5))
    plt.plot(summary_df["Depth"], summary_df["Train_F1"], "o--", label="Train F1")
    plt.plot(summary_df["Depth"], summary_df["Test_F1"], "s-", label="Test F1")
    plt.xlabel("Network Depth (Number of Hidden Layers)")
    plt.ylabel("Average F1 Score")
    plt.title("Average F1 Score vs Network Depth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "f1_vs_depth.png"))
    plt.close()

    print("\nAll results, metrics, and plots saved in:", output_folder)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python c.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)

    train_path, test_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    main(train_path, test_path, output_folder)
