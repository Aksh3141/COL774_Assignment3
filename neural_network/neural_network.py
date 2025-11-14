import numpy as np
from typing import List, Optional, Tuple

EPS = 1e-12 

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    m = y.shape[0]
    oh = np.zeros((m, n_classes), dtype=np.float32)
    oh[np.arange(m), y.astype(int)] = 1.0
    return oh

class NeuralNet:
    def __init__(
        self, hidden_layers: List[int], input_dim: Optional[int] = None,
        n_classes: Optional[int] = None, lr: float = 0.1,
        batch_size: int = 64, epochs: int = 50, seed: Optional[int] = 42,
        weight_scale: Optional[float] = None, l2: float = 0.0, verbose: bool = True,
    ):
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
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        np.clip(z, -50, 50, out=z)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_grad(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_max = np.max(z, axis=1, keepdims=True)
        np.subtract(z, z_max, out=z)
        np.exp(z, out=z)
        z_sum = np.sum(z, axis=1, keepdims=True)
        z /= (z_sum + EPS)
        return z

    def _init_params(self, input_dim: int, n_classes: int):
        layer_dims = [input_dim] + self.hidden_layers + [n_classes]
        self.weights, self.biases = [], []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            scale = self.weight_scale or np.sqrt(2.0 / (in_dim + out_dim))
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
                a = self._sigmoid(z)
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
                delta = (delta @ self.weights[l].T) * self._sigmoid_grad(activations[l])
        return dWs, dbs

    def fit(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-5):
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
