import os
import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support

def load_dataset(folder_path, img_size=(32, 32)):
    X, y = [], []
    class_names = sorted(os.listdir(folder_path))
    for label, class_folder in enumerate(class_names):
        class_dir = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img.flatten() / 255.0)
            y.append(label)
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
    return per_class_df, avg_precision, avg_recall, avg_f1 , y_pred

def main(train_path, test_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    np.random.seed(42)

    hidden_layer_configs = [
        [512],
        [512, 256],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]

    X_train, y_train, class_names = load_dataset(train_path)
    X_test, y_test, _ = load_dataset(test_path)

    summary_records = []
    all_predictions = []
    for i, hidden_layers in enumerate(hidden_layer_configs, start=1):
        print(f"\nTraining MLPClassifier with hidden layers = {hidden_layers}")

        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation='relu',
            solver='sgd',
            alpha=0.0,
            batch_size=32,
            learning_rate='constant',
            max_iter=300,
            early_stopping=True,   
            n_iter_no_change=5,
            tol=1e-3,
            random_state=42,
            verbose=True,
        )

        model.fit(X_train, y_train)

        # Evaluate
        _, train_prec, train_rec, train_f1,_ = evaluate_model(model, X_train, y_train, class_names)
        _, test_prec, test_rec, test_f1,y_pred_test = evaluate_model(model, X_test, y_test, class_names)
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
        
    pred_df = pd.DataFrame(all_predictions, columns=["Prediction"])
    pred_df.to_csv(os.path.join(output_folder, "prediction_d.csv"), index=False)
    summary_df = pd.DataFrame(summary_records)
    # summary_df.to_csv(os.path.join(output_folder, "summary_e.csv"), index=False)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(summary_df["Depth"], summary_df["Train_F1"], "o--", label="Train F1")
    plt.plot(summary_df["Depth"], summary_df["Test_F1"], "s-", label="Test F1")
    plt.xlabel("Network Depth (Hidden Layers)")
    plt.ylabel("Average F1 Score")
    plt.title("Average F1 Score vs Network Depth (MLPClassifier with ReLU)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "f1_vs_depth_mlp.png"))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python e.py <train_data_path> <test_data_path> <output_folder>")
        sys.exit(1)

    train_path, test_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    main(train_path, test_path, output_folder)
