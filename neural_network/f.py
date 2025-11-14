import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle

class NeuralNetReLU:
    def __init__(self, hidden_layers, input_dim, n_classes,
                 lr=0.001, batch_size=64, epochs=20, l2=0.0, verbose=True):
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose
        self.weights, self.biases = self._initialize_weights()

    def _initialize_weights(self):
        layer_sizes = [self.input_dim] + self.hidden_layers + [self.n_classes]
        weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
                   for i in range(len(layer_sizes) - 1)]
        biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
        return weights, biases

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations, pre_activations = [X], []
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            activations.append(self.relu(z))
        z_final = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z_final)
        activations.append(self.softmax(z_final))
        return activations, pre_activations

    def backward(self, X, y_true, activations, pre_activations):
        m = X.shape[0]
        grads_w, grads_b = [], []
        y_pred = activations[-1]
        y_true_oh = np.zeros_like(y_pred)
        y_true_oh[np.arange(m), y_true] = 1
        dz = (y_pred - y_true_oh) / m

        for i in reversed(range(len(self.weights))):
            dw = np.dot(activations[i].T, dz) + self.l2 * self.weights[i]
            db = np.sum(dz, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i != 0:
                dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(pre_activations[i - 1])

        return grads_w, grads_b

    def update(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            if self.biases[i].shape != grads_b[i].shape:
                grads_b[i] = grads_b[i].reshape(self.biases[i].shape)
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]


    # ---------------- Training Loop ----------------
    def fit(self, X_train, y_train, X_test, y_test):
        n = X_train.shape[0]
        train_f1s, test_f1s = [], []
        for epoch in range(self.epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for i in range(0, n, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]
                activations, pre_activations = self.forward(X_batch)
                grads_w, grads_b = self.backward(X_batch, y_batch, activations, pre_activations)
                self.update(grads_w, grads_b)

            y_pred_train = self.predict(X_train)
            y_pred_test = self.predict(X_test)
            _, _, f1_train, _ = precision_recall_fscore_support(
                y_train, y_pred_train, average='macro', zero_division=0)
            _, _, f1_test, _ = precision_recall_fscore_support(
                y_test, y_pred_test, average='macro', zero_division=0)
            train_f1s.append(f1_train)
            test_f1s.append(f1_test)
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Train F1: {f1_train:.3f}, Test F1: {f1_test:.3f}")
        return train_f1s, test_f1s


    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def save(self, filename):
        np.savez(filename,
                 weights=np.array(self.weights, dtype=object),
                 biases=np.array(self.biases, dtype=object),
                 hidden_layers=np.array(self.hidden_layers, dtype=object),
                 input_dim=self.input_dim,
                 n_classes=self.n_classes)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        data = np.load(filename, allow_pickle=True)
        model = cls(list(data["hidden_layers"]),
                    int(data["input_dim"]),
                    int(data["n_classes"]))
        model.weights = list(data["weights"])
        # Ensure all biases are 2D (1, n_units)
        model.biases = [b.reshape(1, -1) if b.ndim == 1 else b for b in data["biases"]]
        print(f"Model loaded from {filename}")
        return model



# B. Dataset Loading (using cv2, RGB)
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



# Main Script
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python f.py <train_data_path> <test_data_path> <output_folder>")
        sys.exit(1)

    train_path, test_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load datasets
    X_train, y_train, _ = load_dataset(train_path)
    X_test, y_test, _ = load_dataset(test_path)
    input_dim = X_train.shape[1]
    n_classes = 10

    # Train from scratch 
    model_scratch = NeuralNetReLU([512, 256, 128, 64],
                                  input_dim, n_classes,
                                  lr=0.001, epochs=20)
    f1_train_scratch, f1_test_scratch = model_scratch.fit(X_train, y_train, X_test, y_test)
    preds_scratch = model_scratch.predict(X_test)


    # Fine-tuning (Transfer Learning)
    consonant_model = NeuralNetReLU.load("model_depth_4.npz")

    # Create a new model with same architecture
    model_tl = NeuralNetReLU(consonant_model.hidden_layers,
                             consonant_model.input_dim,
                             n_classes,
                             lr=0.001, epochs=20)

    # Load pretrained weights & biases
    import copy
    model_tl.weights[:-1] = [w.copy() for w in consonant_model.weights[:-1]]
    model_tl.biases[:-1] = [b.copy() for b in consonant_model.biases[:-1]]

    # Train entire model (fine-tuning)
    f1_train_tl, f1_test_tl = model_tl.fit(X_train, y_train, X_test, y_test)
    preds_tl = model_tl.predict(X_test)


    # Save predictions
    all_preds = []
    all_preds.extend((preds_scratch+1).tolist())
    all_preds.extend((preds_tl+1).tolist())
    pred_path = os.path.join(output_folder, "prediction_f.csv")
    pd.DataFrame({"predictions": all_preds}).to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Plot F1-scores
    plt.figure(figsize=(8, 5))
    plt.plot(f1_train_scratch, label="Train F1 - Scratch", linestyle='--')
    plt.plot(f1_test_scratch, label="Test F1 - Scratch")
    plt.plot(f1_train_tl, label="Train F1 - Fine-tuned", linestyle='--')
    plt.plot(f1_test_tl, label="Test F1 - Fine-tuned")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Digits Classification: Scratch vs Fine-tuned (Full Training)")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_folder, "comparison_plot.png")
    plt.savefig(plot_path)
    #plt.show()
    print(f"Saved F1 plot to {plot_path}")
