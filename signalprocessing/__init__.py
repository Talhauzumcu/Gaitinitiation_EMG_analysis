"""
Signal Processing Package

A collection of signal processing utilities for EMG analysis including:
- Filters (low_pass, high_pass, band_pass, band_stop, notch)
- General operations (moving_avg, downsample, upsample, etc.)
- Arithmetics (add, subtract, multiply, etc.)
- Math functions (abs_value, rectify, sqrt, etc.)
- Spectral analysis (fft, power_spectrum, etc.)
- Statistics (mean, std, variance, etc.)
- Time-frequency analysis
- Synchronization utilities
"""

# Import submodules for access like: sp.arithmetics.add_signals(...)
from . import arithmetics
from . import filters
from . import general
from . import math_functions
from . import spectral_analysis
from . import stats
from . import stats_curve
from . import time_frequency

__version__ = "1.0.0"
__all__ = [
    # Submodules
    'arithmetics', 'filters', 'general', 'math_functions', 
    'spectral_analysis', 'stats', 'stats_curve', 'synchronization', 'time_frequency',
]
