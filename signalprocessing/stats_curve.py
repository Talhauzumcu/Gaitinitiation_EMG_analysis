import numpy as np
import scipy.stats as st
from scipy.signal import correlate as cross_corr


def mean_curve(signals: list[np.ndarray]) -> np.ndarray:
    """
    Computes the mean curve from multiple signals.

    Args:
        signals (list[np.ndarray]): List of signal arrays.

    Returns:
        np.ndarray: Mean signal array.
    """
    signals_array = np.array(signals)
    mean = np.mean(signals_array, axis=0)
    return mean

def median_curve(signals: list[np.ndarray]) -> np.ndarray:
    """
    Computes the median curve from multiple signals.

    Args:
        signals (list[np.ndarray]): List of signal arrays.

    Returns:
        np.ndarray: Median signal array.
    """
    signals_array = np.array(signals)
    median = np.median(signals_array, axis=0)
    return median

def histogram(sig: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes histogram of data.

    Args:
        sig (np.ndarray): Input data array.
        bins (int, optional): Number of bins. Defaults to 10.

    Returns:
        tuple[np.ndarray, np.ndarray]: Bin counts and bin edges.
    """
    counts, edges = np.histogram(sig, bins=bins)
    return counts, edges

def confidence_interval(sig: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes confidence interval for mean.

    Args:
        sig (np.ndarray): Input data array.
        confidence (float, optional): Confidence level (0-1). Defaults to 0.95.

    Returns:
        tuple[float, float]: Lower and upper bounds of the confidence interval.
    """
    N = len(sig)
    mean = np.mean(sig)
    sem = np.std(sig) / np.sqrt(N)
    df = N - 1
    alpha = 1 - confidence

    if N < 30: # Use t-distribution if N < 30
        crit_val = st.t.ppf(1 - alpha/2, df)
    else:
        crit_val = st.norm.ppf(1 - alpha/2)
    
    margin = crit_val * sem
    return float(mean - margin), float(mean + margin)

def correlation(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Computes Pearson correlation coefficient.

    Args:
        sig1 (np.ndarray): First signal array.
        sig2 (np.ndarray): Second signal array.

    Returns:
        float: Correlation coefficient.
    """
    return float(np.corrcoef(sig1, sig2)[0,1])

def cross_correlation(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """
    Computes cross-correlation between signals.

    Args:
        signal1 (np.ndarray): First signal array.
        signal2 (np.ndarray): Second signal array.

    Returns:
        np.ndarray: Cross-correlation array.
    """
    xcorr = cross_corr(signal1, signal2, mode='full')
    return xcorr