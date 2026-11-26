# Learning_representations_by_backpropagating_errors
import numpy as np

class MLP:
    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def linear(self, x: np.ndarray, w: np.ndarray , b: np.ndarray) -> np.ndarray:
        return x @ w + b

    def activation(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def loss(self, a: np.ndarray, y_true: np.ndarray) -> np.float64:
        return np.mean(np.square(a - y_true))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = self.linear(x, self.w1, self.b1)
        self.a1 = self.activation(self.z1)
        self.z2 = self.linear(self.a1, self.w2, self.b2)
        self.a2 = self.activation(self.z2)
        return self.a2

    def backward(self, y_true: np.ndarray):
        dL_da2 = 2 * (self.a2 - y_true) / self.a2.size
        da_dz2 = 1 - np.square(self.a2)
        dL_dz2 = dL_da2 * da_dz2
        dW2 = self.a1.T @ dL_dz2
        db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ self.w2.T
        da1_dz1 = 1 - np.square(self.a1)
        dL_dz1 = dL_da1 * da1_dz1
        dW1 = self.x.T @ dL_dz1
        db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def step(self, y_true: np.ndarray, lr=0.1) -> None:
        dW1, db1, dW2, db2 = self.backward(y_true)
        self.w1 -= lr * dW1
        self.b1 -= lr * db1
        self.w2 -= lr * dW2
        self.b2 -= lr * db2

#def grad_check(y: np.ndarray, d = np.ndarray) -> np.float64:
#    pass

def grad_check(model: MLP, x: np.ndarray, y_true: np.ndarray, eps: float = 1e-6) -> bool:
    x = np.asarray(x, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    for name in ("w1", "b1", "w2", "b2"):
        setattr(model, name, np.asarray(getattr(model, name), dtype=np.float64))

    model.forward(x)
    dW1, db1, dW2, db2 = model.backward(y_true)

    params_and_grads = [
        ("w1", model.w1, dW1),
        ("b1", model.b1, db1),
        ("w2", model.w2, dW2),
        ("b2", model.b2, db2),
    ]

    all_passed = True

    for name, param, grad_ana in params_and_grads:
        grad_num = np.zeros_like(param, dtype=np.float64)

        for idx in np.ndindex(param.shape):
            orig = param[idx]

            param[idx] = orig + eps
            loss_plus = model.loss(model.forward(x), y_true)

            param[idx] = orig - eps
            loss_minus = model.loss(model.forward(x), y_true)

            param[idx] = orig
            grad_num[idx] = (loss_plus - loss_minus) / (2.0 * eps)

        diff = np.linalg.norm(grad_ana - grad_num)
        denom = max(np.linalg.norm(grad_ana), np.linalg.norm(grad_num))
        rel_error = diff / denom if denom > 0 else 0.0

        passed = rel_error < 1e-5
        all_passed = all_passed and passed
        print(f"{name}: relative error = {rel_error:.3e} ({'PASS' if passed else 'FAIL'})")

    model.forward(x)
    return all_passed

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[-1], [1], [1], [-1]], dtype=np.float64)

    mlp = MLP(
        w1=rng.normal(0, 0.5, size=(2, 4)),
        b1=np.zeros((1, 4), dtype=np.float64),
        w2=rng.normal(0, 0.5, size=(4, 1)),
        b2=np.zeros((1, 1), dtype=np.float64),
    )

    grad_check(mlp, X, y)
