import numpy as np
import pywt


def cwt(signal: np.ndarray, fs: float, n_scales: int = 128,
    f_min: float = None, f_max: float = None, norm: str = 'absolute', spektrum: str = 'power', wavelet: str = 'cmor',
    bandwidth: str = '1.5', center_frequency: str = '2.0') -> tuple[np.ndarray, np.ndarray]:
    """Calculates the continous wavelet transform of a signal using PyWavelets.
    
    Args:
        signal (np.ndarray): Input signal array
        fs (float): Sampling rate in Hz
        n_scales (int, optional): Number of scales. Defaults to 128.
        f_min (float, optional): Lower frequency bound in Hz. Defaults to None.
        f_max (float, optional): Upper frequency bound in Hz. Defaults to None.
        norm (str, optional): Normalization type ('absolute' or 'relative'). Defaults to 'absolute'.
        spektrum (str, optional): Type of spectrum ('power' or 'magnitude'). Defaults to 'power'.
        wavelet (str, optional): Wavelet type. Defaults to 'cmor'. ('cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan')
        bandwidth (str, optional): Bandwidth of the wavelet. Defaults to '1.5'.
        center_frequency (str, optional): Center frequency of the wavelet. Defaults to '2.0'.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            - power/amplitude: Time-frequency coeff matrix
            - freqs: Frequency array
    """

    if f_min is None:
        f_min = 1 / (len(signal) / fs)
    if f_max is None:
        f_max = fs / 2

    wavelet = f'{wavelet}{bandwidth}-{center_frequency}'
    freqs = np.geomspace(f_min, f_max, n_scales)
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq / (freqs * (1/fs))
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    magnitude = np.abs(coeffs) 
    power = magnitude ** 2
    if spektrum == 'power':
        data = power
    elif spektrum == 'magnitude':
        data = magnitude

    if norm == 'relative':
        data /= np.max(data)
    
    return data, freqs

def frequency_ridge(signal: np.ndarray, fs: float, n_scales: int = 128,
    f_min: float = None, f_max: float = None, norm: str = 'absolute', spektrum: str = 'power', wavelet: str = 'cmor',
    bandwidth: str = '1.5', center_frequency: str = '2.0') -> np.ndarray:
    """Calculates the frequency ridge of a signal using the continous wavelet transform.
    
    Args:
        signal (np.ndarray): Input signal array
        fs (float): Sampling rate in Hz
        n_scales (int, optional): Number of scales. Defaults to 128.
        f_min (float, optional): Lower frequency bound in Hz. Defaults to None.
        f_max (float, optional): Upper frequency bound in Hz. Defaults to None.
        norm (str, optional): Normalization type ('absolute' or 'relative'). Defaults to 'absolute'.
        spektrum (str, optional): Type of spectrum ('power' or 'magnitude'). Defaults to 'power'.
        wavelet (str, optional): Wavelet type. Defaults to 'cmor'. ('cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan')
        bandwidth (str, optional): Bandwidth of the wavelet. Defaults to '1.5'.
        center_frequency (str, optional): Center frequency of the wavelet. Defaults to '2.0'.
        
    Returns:
        np.ndarray: Ridge frequency array
    """
    
    power, freqs = cwt(signal, fs, n_scales, f_min, f_max, norm, spektrum, wavelet, bandwidth, center_frequency)
    ridge_freq = np.array([freqs[np.argmax(power[:, i])] for i in range(power.shape[1])])
    
    return ridge_freq


