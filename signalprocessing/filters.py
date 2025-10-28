import numpy as np
from scipy.signal import butter, filtfilt


def low_pass(signal: np.ndarray, cutoff_freq: float, sampling_rate: float, order: float = 4) -> np.ndarray:
    """
    Low pass filter implementation.

    Args:
        signal (np.ndarray): Input signal array.
        cutoff_freq (float): Cutoff frequency in Hz.
        sampling_rate (float): Sampling rate in Hz.
        order (float, optional): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal array.
    """
    nyquist = sampling_rate / 2
    normalized_cutoff_freq = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff_freq, btype='low')
    return filtfilt(b, a, signal)

def high_pass(signal: np.ndarray, cutoff_freq: float, sampling_rate: float, order: float = 4) -> np.ndarray:
    """
    High pass filter implementation.

    Args:
        signal (np.ndarray): Input signal array.
        cutoff_freq (float): Cutoff frequency in Hz.
        sampling_rate (float): Sampling rate in Hz.
        order (float, optional): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal array.
    """
    nyquist = sampling_rate / 2
    normalized_cutoff_freq = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff_freq, btype='high')
    return filtfilt(b, a, signal)

def bandpass(signal: np.ndarray, low_freq: float, high_freq: float, sampling_rate: float, order: float = 4) -> np.ndarray:
    """
    Bandpass filter implementation.

    Args:
        signal (np.ndarray): Input signal array.
        low_freq (float): Lower cutoff frequency in Hz.
        high_freq (float): Higher cutoff frequency in Hz.
        sampling_rate (float): Sampling rate in Hz.
        order (float, optional): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal array.
    """
    nyquist = sampling_rate / 2
    normalized_freqs = [low_freq/nyquist, high_freq/nyquist]
    b, a = butter(order, normalized_freqs, btype='band')
    return filtfilt(b, a, signal)

def band_block(signal: np.ndarray, low_freq: float, high_freq: float, sampling_rate: float, order: float = 4) -> np.ndarray:
    """
    Band block filter implementation.

    Args:
        signal (np.ndarray): Input signal array.
        low_freq (float): Lower cutoff frequency in Hz.
        high_freq (float): Higher cutoff frequency in Hz.
        sampling_rate (float): Sampling rate in Hz.
        order (float, optional): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal array.
    """
    nyquist = sampling_rate / 2
    normalized_freqs = [low_freq/nyquist, high_freq/nyquist]
    b, a = butter(order, normalized_freqs, btype='stop')
    return filtfilt(b, a, signal)

