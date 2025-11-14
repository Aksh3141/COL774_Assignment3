import numpy as np
from typing import List, Optional, Tuple
import os
import sys
import cv2
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt 


EPS = 1e-12

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    m = y.shape[0]
    oh = np.zeros((m, n_classes), dtype=np.float32)
    oh[np.arange(m), y.astype(int)] = 1.0
    return oh


class NeuralNetReLU:
    def __init__(
        self, hidden_layers: List[int], input_dim: Optional[int] = None,
        n_classes: Optional[int] = None, lr: float = 0.1,
        batch_size: int = 64, epochs: int = 50, seed: Optional[int] = 42,
        weight_scale: Optional[float] = None, l2: float = 0.0, verbose: bool = True,):

        self.hidden_layers = list(hidden_layers)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = seed
        self.weight_scale = weight_scale
        self.l2 = float(l2)
        self.verbose = verbose

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        if seed is not None:
            np.random.seed(seed)

    # --- Activations ---
    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        # Sub-gradient: derivative is 1 for z>0, 0 for z<0, and random(0,1) for z=0 (optional)
        grad = np.zeros_like(z, dtype=np.float32)
        grad[z > 0] = 1.0
        return grad

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_max = np.max(z, axis=1, keepdims=True)
        z = z - z_max
        np.exp(z, out=z)
        z_sum = np.sum(z, axis=1, keepdims=True)
        z /= (z_sum + EPS)
        return z

    def _init_params(self, input_dim: int, n_classes: int):
        layer_dims = [input_dim] + self.hidden_layers + [n_classes]
        self.weights, self.biases = [], []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            scale = self.weight_scale or np.sqrt(2.0 / in_dim)
            self.weights.append((np.random.randn(in_dim, out_dim) * scale).astype(np.float32))
            self.biases.append(np.zeros(out_dim, dtype=np.float32))
        self.input_dim, self.n_classes = input_dim, n_classes

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = X
        activations = [a]
        zs = [None]
        for i, (W, b) in enumerate(zip(self.weights, self.biases), start=1):
            z = a @ W + b
            if i < len(self.weights):
                a = self._relu(z)
            else:
                a = self._softmax(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def _cross_entropy_loss(self, y_hat: np.ndarray, y_onehot: np.ndarray) -> float:
        m = y_hat.shape[0]
        np.clip(y_hat, EPS, 1.0 - EPS, out=y_hat)
        loss = -np.sum(y_onehot * np.log(y_hat)) / m
        if self.l2 > 0:
            loss += 0.5 * self.l2 * sum(np.sum(W * W) for W in self.weights)
        return loss

    def _backprop(self, activations, zs, y_onehot):
        m = y_onehot.shape[0]
        L = len(self.weights)
        dWs = [None] * L
        dbs = [None] * L

        delta = (activations[-1] - y_onehot) / m  

        for l in reversed(range(L)):
            a_prev = activations[l]
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0)
            if self.l2 > 0:
                dW += self.l2 * self.weights[l]
            dWs[l], dbs[l] = dW, db

            if l > 0:
                # Apply ReLU gradient (use Zs here, not activations)
                delta = (delta @ self.weights[l].T) * self._relu_grad(zs[l])
        return dWs, dbs

    def fit(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-4):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        N = X.shape[0]
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.n_classes is None:
            self.n_classes = int(y.max() + 1)
        if not self.weights:
            self._init_params(self.input_dim, self.n_classes)

        y_onehot_full = one_hot(y, self.n_classes)
        prev_loss = np.inf

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(N)
            X_sh, y_sh = X[perm], y_onehot_full[perm]
            epoch_loss = 0.0

            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                Xb, yb = X_sh[start:end], y_sh[start:end]
                activations, zs = self._forward(Xb)
                loss = self._cross_entropy_loss(activations[-1], yb)
                epoch_loss += loss * Xb.shape[0]
                dWs, dbs = self._backprop(activations, zs, yb)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * dWs[i]
                    self.biases[i] -= self.lr * dbs[i]

            epoch_loss /= N
            change = abs(prev_loss - epoch_loss)

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == 1):
                acc = self.score(X, y)
                print(f"Epoch {epoch:3d}/{self.epochs} - loss: {epoch_loss:.6f} - loss: {change:.2e} - acc: {acc:.4f}")

            if change < tol:
                if self.verbose:
                    print(f"Stopping early at epoch {epoch}")
                break
            prev_loss = epoch_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

    def save(self, filename: str):
        """Save the trained model to an NPZ file (same folder as code)."""
        np.savez(
            filename,
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object),
            hidden_layers=np.array(self.hidden_layers, dtype=object),
            input_dim=self.input_dim,
            n_classes=self.n_classes,
            lr=self.lr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            l2=self.l2,
        )
        if self.verbose:
            print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename: str, verbose: bool = False):
        """Load a saved NPZ model file and return a NeuralNetReLU instance."""
        data = np.load(filename, allow_pickle=True)
        model = cls(
            hidden_layers=list(data["hidden_layers"]),
            input_dim=int(data["input_dim"]),
            n_classes=int(data["n_classes"]),
            lr=float(data["lr"]),
            batch_size=int(data["batch_size"]),
            epochs=int(data["epochs"]),
            l2=float(data["l2"]),
            verbose=verbose
        )
        model.weights = list(data["weights"])
        model.biases = list(data["biases"])
        if verbose:
            print(f"Model loaded from {filename}")
        return model

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

    # Hidden layer configurations
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
        print(f"Training model with depth={i}, hidden layers={hidden_layers} (ReLU)")
        print("---------------------------------------------")

        model = NeuralNetReLU(
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
        model.save(f"model_depth_{i}.npz")

    
    
    # --- Save predictions ---
    pred_df = pd.DataFrame(all_predictions, columns=["Prediction"])
    pred_df.to_csv(os.path.join(output_folder, "prediction_d.csv"), index=False)

    # --- Save summary metrics ---
    summary_df = pd.DataFrame(summary_records)
    # summary_df.to_csv(os.path.join(output_folder, "summary_d.csv"), index=False)

    # --- Plot F1 vs depth ---
    plt.figure(figsize=(7, 5))
    plt.plot(summary_df["Depth"], summary_df["Train_F1"], "o--", label="Train F1 (ReLU)")
    plt.plot(summary_df["Depth"], summary_df["Test_F1"], "s-", label="Test F1 (ReLU)")
    plt.xlabel("Network Depth (Number of Hidden Layers)")
    plt.ylabel("Average F1 Score")
    plt.title("Average F1 Score vs Network Depth (ReLU Activation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "f1_vs_depth_relu.png"))
    plt.close()

    print("\nAll results, metrics, and plots saved in:", output_folder)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python d.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)

    train_path, test_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    main(train_path, test_path, output_folder)
