import numpy as np

def sin_signal(sig: np.ndarray) -> np.ndarray:
    return np.sin(sig)

def cos_signal(sig: np.ndarray) -> np.ndarray:
    return np.cos(sig)

def log10_signal(sig: np.ndarray) -> np.ndarray:
    return np.log10(sig)

def ln_signal(sig: np.ndarray) -> np.ndarray:
    return np.log(sig)

def inverse_signal(sig: np.ndarray) -> np.ndarray:
    return np.divide(1, sig)

def sqrt_signal(sig: np.ndarray) -> np.ndarray:
    return np.sqrt(sig)


