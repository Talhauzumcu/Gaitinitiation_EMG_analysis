import numpy as np

def multiply_constant(sig: np.ndarray, constant: float) -> np.ndarray:
    return sig * constant

def add_constant(sig: np.ndarray, constant: float) -> np.ndarray:
    return sig + constant

def add_signals(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    return sig1 + sig2

def subtract_signals(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    return sig1 - sig2

def multiply_signals(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    return sig1 * sig2

def divide_signals(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    return np.divide(sig1, sig2)

def abs_signal(sig: np.ndarray) -> np.ndarray:
    return np.abs(sig)

def power_signal(sig: np.ndarray, power: float) -> np.ndarray:
    return np.power(sig, power)
