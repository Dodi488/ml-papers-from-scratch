# Adam:_A_Method_for_Stochastic_Optimization

import numpy as np
from backprop import MLP, grad_check

class Adam:
    def __init__(self, theta, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.theta = theta
        self.m_0 = [np.zeros_like(p) for p in theta]
        self.v_0 = [np.zeros_like(p) for p in theta]
        self.t = 0
        self.m_hat = None
        self.v_hat = None

    def optimize(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.theta, grads)):
            self.m_0[i] = (self.beta1 * self.m_0[i]) + (1 - self.beta1) * grad
            self.v_0[i] = (self.beta2 * self.v_0[i]) + (1 - self.beta2) * np.square(grad)
            self.m_hat = self.m_0[i] / (1 - self.beta1 ** self.t)
            self.v_hat = self.v_0[i] / (1 - self.beta2 ** self.t)
            param -= self.alpha * (self.m_hat / (np.sqrt(self.v_hat) + self.epsilon))

        return self.theta

rng = np.random.default_rng(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[-1], [1], [1], [-1]], dtype=np.float64)

mlp = MLP(w1=rng.normal(0, 0.5, size=(2, 4)), b1=np.zeros((1, 4), dtype=np.float64), w2=rng.normal(0, 0.5, size=(4, 1)), b2=np.zeros((1, 1), dtype=np.float64),)
optimizer = Adam([mlp.w1, mlp.b1, mlp.w2, mlp.b2])

for step in range(1, 2001):
    preds = mlp.forward(X)
    loss = mlp.loss(preds, y)
    grads = mlp.backward(y)
    optimizer.optimize(grads)

    if step % 500 == 0:
        print(f"Step {step:4d} | Loss: {loss:.6f}")

print("Final predictions:\n", np.round(mlp.forward(X), 3))
