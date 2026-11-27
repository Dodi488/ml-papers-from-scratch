# Understanding_the_Difficulty_of_Training_Deep_Feedforward_Neural_Networks
import numpy as np
from backprop import MLP

def xavier_init(W: np.ndarray, fan_in: int, fan_out: int) -> np.ndarray:
    L = np.sqrt(6 / (fan_in + fan_out))
    return W * (2 * L) - L
