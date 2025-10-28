import numpy as np
from scipy import stats as st


def mean(sig: np.ndarray) -> float:
    return float(np.mean(sig))

def median(sig: np.ndarray) -> float:
    return float(np.median(sig))

def std(sig: np.ndarray) -> float:
    return float(np.std(sig))

def sem(sig: np.ndarray) -> float:
    return float(np.std(sig) / np.sqrt(len(sig)))

def minimum(sig: np.ndarray) -> float:
    return float(np.min(sig))

def maximum(sig: np.ndarray) -> float:
    return float(np.max(sig))

def variance(sig: np.ndarray) -> float:
    return float(np.var(sig))

def skewness(sig: np.ndarray) -> float:
    return float(st.skew(sig))

def kurtosis(sig: np.ndarray) -> float:
    return float(st.kurtosis(sig))

def is_normal(sig: np.ndarray, alpha: float = 0.05) -> bool:
    _, p_value = st.normaltest(sig)
    return p_value > alpha


