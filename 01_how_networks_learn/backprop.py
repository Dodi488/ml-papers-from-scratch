# Learning_representations_by_backpropagating_errors
import numpy as np

class MLP:
    def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
        self.w = w
        self.b = b
        self.z = None
        self.a = None

    def linear(self, x, w, b) -> np.ndarray:
        return x @ w + b

    def activation(self, z) -> np.ndarray:
        return np.tanh(z)

    def loss(self, a: np.ndarray, y_true: np.ndarray) -> np.float64:
        return np.mean(np.square(a - y_true))

    def forward(self, x) -> np.ndarray:
        self.x = x
        self.z = self.linear(x, self.w, self.b)
        self.a = self.activation(self.z)
        return self.a

    def backward(self, y_true):
        dL_da = 2 * (self.a - y_true) / self.a.size
        da_dz = 1 - np.square(self.a)
        dL_dz = dL_da * da_dz
        dW = self.x.T @ dL_dz
        db = np.sum(dL_dz, axis=0, keepdims=True)
        return dW, db

    def step(self, y_true, lr=0.1) -> None:
        dW, db = self.backward(y_true)
        self.w -= lr * dW
        self.b -= lr * db
        

def grad_check(y: np.ndarray, d = np.ndarray) -> np.float64:
    pass
