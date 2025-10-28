import numpy as np
from scipy.signal import hilbert

def amplitude_spectrum(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the amplitude spectrum of a signal.

    Args:
        signal (np.ndarray): Input signal array.
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frequency array and Amplitude array.
    """
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/sampling_rate)
    fft = np.fft.fft(signal)
    amplitude = np.abs(fft) * 2 / n
    return freqs, amplitude

def power_spectrum(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectrum of a signal.

    Args:
        signal (np.ndarray): Input signal array.
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frequency array and Power spectrum array.
    """
    freqs, amplitude = amplitude_spectrum(signal, sampling_rate)
    power = amplitude**2
    return freqs, power

def phase_spectrum(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate phase spectrum of a signal.
    
    Args:
        signal (np.ndarray): Input signal array.
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frequency array and Phase spectrum array.
    """
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/sampling_rate)
    fft = np.fft.fft(signal)
    phase = np.angle(fft)
    return freqs, phase

def hilbert_transform(signal: np.ndarray) -> np.ndarray:
    """
    Calculate Hilbert transform of a signal.

    Args:
        signal (np.ndarray): Input signal array.

    Returns:
        np.ndarray: Hilbert transformed signal.
    """
    analytic_signal = hilbert(signal)
    return np.imag(analytic_signal)

def analytic_signal(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate analytic signal of a signal.

    Args:
        signal (np.ndarray): Input signal array.

    Returns:
        tuple[np.ndarray, np.ndarray]: Real part of the analytic signal and Hilbert transformed signal.
    """
    analytic_signal = hilbert(signal)
    real = np.real(analytic_signal)
    imag = np.imag(analytic_signal)
    return real, imag

def instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """
    Calculate instantaneous phase of a signal.

    Args:
        signal (np.ndarray): Input signal array.

    Returns:
        np.ndarray: Instantaneous phase array.
    """
    analytic = analytic_signal(signal)
    analytic = analytic[0] + 1j * analytic[1]
    return np.angle(analytic)

def instantaneous_frequency(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Calculate instantaneous frequency of a signal.

    Args:
        signal (np.ndarray): Input signal array.
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        np.ndarray: Instantaneous frequency array.
    """
    phase = instantaneous_phase(signal)
    unwrapped_phase = np.unwrap(phase)
    freq = np.diff(unwrapped_phase) * sampling_rate / (2 * np.pi)
    return freq