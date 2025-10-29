import numpy as np
from scipy.signal import decimate,hilbert
from scipy import interpolate as interp

def moving_avg(sig1: np.ndarray, window_size: int, mode: str = 'same') -> np.ndarray:
    """
    Apply moving average filter to signal.

    Args:
        sig1 (np.ndarray): Input signal as numpy array.
        window_size (int): Size of moving average window.
        mode (str): Padding mode ('full', 'same', or 'valid').

    Returns:
        np.ndarray: Filtered signal.
    """
    kernel = np.ones(window_size) / window_size
    filtered = np.convolve(sig1, kernel, mode=mode)
    return filtered

def downsample(sig1: np.ndarray, factor: int, ftype: str = 'iir') -> np.ndarray:
    """
    Subsample signal by decimation with anti-aliasing filter.

    Args:
        sig1 (np.ndarray): Input signal.
        factor (int): Subsampling factor.
        ftype (str): Anti-aliasing filter type ('iir', 'fir').

    Returns:
        np.ndarray: Subsampled signal.
    """
    return decimate(sig1, factor, ftype=ftype)

def interpolate(sig1: np.ndarray, fs_current: float, fs_desired: float, method: str = 'linear') -> np.ndarray:
    """
    Interpolate signal to a new sampling frequency.

    Args:
        sig1 (np.ndarray): Input signal.
        fs_current (float): Current sampling frequency in Hz.
        fs_desired (float): Desired sampling frequency in Hz.
        method (str): Interpolation method ('linear', 'nearest', 'cubic', etc.).

    Returns:
        np.ndarray: Interpolated signal.
    """
    t_current = np.arange(len(sig1)) / fs_current
    new_length = int(len(sig1) * fs_desired / fs_current)
    t_desired = np.arange(new_length) / fs_desired
    f = interp.interp1d(t_current, sig1, kind=method)
    return f(t_desired)

def envelope(sig1: np.ndarray) -> np.ndarray:
    """
    Calculate signal envelope using Hilbert transform.

    Args:
        sig1 (np.ndarray): Input signal.

    Returns:
        np.ndarray: Signal envelope.
    """
    envelope = np.abs(hilbert(sig1))
    return envelope

def remove_dc(sig1: np.ndarray) -> np.ndarray:
    """
    Remove DC component from signal.

    Args:
        sig1 (np.ndarray): Input signal.

    Returns:
        np.ndarray: Signal with DC removed.
    """
    return sig1 - np.mean(sig1)

def remove_linear_drift(sig1: np.ndarray) -> np.ndarray:
    """
    Remove linear drift from signal.

    Args:
        sig1 (np.ndarray): Input signal.

    Returns:
        np.ndarray: Signal with linear drift removed.
    """
    x = np.arange(len(sig1))
    z = np.polyfit(x, sig1, 1)
    drift = np.polyval(z, x)
    return sig1 - drift

def normalize(sig1: np.ndarray) -> np.ndarray:
    """
    Normalize signal to zero mean and unit variance.

    Args:
        sig1 (np.ndarray): Input signal.

    Returns:
        np.ndarray: Normalized signal.
    """
    return (sig1 - np.mean(sig1)) / np.std(sig1)

def clip(sig1: np.ndarray, threshold: float) -> np.ndarray:
    """
    Clip signal values to ±threshold.

    Args:
        sig1 (np.ndarray): Input signal.
        threshold (float): Clipping threshold.

    Returns:
        np.ndarray: Clipped signal.
    """
    return np.clip(sig1, -threshold, threshold)

def wrap(sig1: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Wrap signal values between lower and upper bounds.

    Args:
        sig1 (np.ndarray): Input signal.
        lower (float): Lower bound.
        upper (float): Upper bound.

    Returns:
        np.ndarray: Wrapped signal.
    """
    return ((sig1 - lower) % (upper - lower)) + lower

def mirror(sig1: np.ndarray, mirror_length: int, position: str = 'end') -> np.ndarray:
    """
    Mirror signal at the specified end(s).

    Args:
        sig1 (np.ndarray): Input signal.
        mirror_length (int): Length of the mirrored section.
        position (str): 'start', 'end', or 'both'.

    Returns:
        np.ndarray: Mirrored signal.
    """
    if mirror_length <= 0:
        raise ValueError("mirror_length must be positive.")
    if mirror_length > len(sig1):
        raise ValueError("mirror_length cannot exceed signal length.")
    if position == 'both':
        return np.concatenate((sig1[mirror_length-1::-1], sig1, sig1[-1:-mirror_length-1:-1]))
    elif position == 'start':
        return np.concatenate((sig1[mirror_length-1::-1], sig1))
    elif position == 'end':
        return np.concatenate((sig1, sig1[-1:-mirror_length-1:-1]))
    else:
        raise ValueError("position must be 'start', 'end', or 'both'.")

def append_signals(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    Append two signals.

    Args:
        sig1 (np.ndarray): First input signal.
        sig2 (np.ndarray): Second input signal.

    Returns:
        np.ndarray: Concatenated signal.
    """
    return np.concatenate((sig1, sig2))

def derivative(sig1: np.ndarray, dt: float = 1.0, fs: float = None) -> np.ndarray:
    """
    Calculate the first derivative of a signal.

    Args:
        sig1 (np.ndarray): Input signal.
        dt (float): Time step between samples.
        fs (float, optional): Sampling frequency in Hz; overrides dt if provided.

    Returns:
        np.ndarray: First derivative.
    """
    if fs is not None:
        dt = 1/fs
    return np.gradient(sig1, dt)

def integrate_signal(sig1: np.ndarray, dt: float = 1.0, fs: float = None) -> np.ndarray:
    """
    Calculate the cumulative integral of a signal.

    Args:
        sig1 (np.ndarray): Input signal.
        dt (float): Time step between samples.
        fs (float, optional): Sampling frequency in Hz; overrides dt if provided.

    Returns:
        np.ndarray: Cumulative integral.
    """
    if fs is not None:
        dt = 1/fs
    return np.cumsum(sig1) * dt