"# Gait Initiation EMG Analysis

A Python-based pipeline for analyzing electromyography (EMG) signals during gait initiation tasks, comparing young adults (YA) and older adults (OA) in stop-signal paradigms.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [EMG Signal Preprocessing](#emg-signal-preprocessing)
- [Analyses](#analyses)
  - [EMG Amplitude Analysis](#emg-amplitude-analysis)
  - [EMG On/Off Detection](#emg-onoff-detection)
  - [Muscle Co-contraction Analysis](#muscle-co-contraction-analysis)
- [Key Events and Time Windows](#key-events-and-time-windows)
- [Output Files](#output-files)
- [Visualizations](#visualizations)

---

## Overview

This project analyzes EMG data collected during gait initiation experiments with a stop-signal paradigm. The analysis compares muscle activation patterns between:
- **Young Adults (YA)** vs **Older Adults (OA)**
- **Successful** vs **Unsuccessful** trials (stopping ability)
- **Early** vs **Late** latency conditions

### EMG Channels Analyzed
- Right Tibialis Anterior (`03_ri_tib_ant`)
- Right Soleus (`01_ri_soleus`)
- Right Gastrocnemius Medialis (`02_ri_gastroc_med`)
- Left Tibialis Anterior (`06_le_tib_ant`)
- Left Soleus (`07_le_soleus`)
- Left Gastrocnemius Medialis (`08_le_gastroc_med`)

---

## Project Structure

```
Gaitinitiation_EMG_analysis/
├── analysis.py              # Main analysis script - runs all EMG analyses
├── analysis_functions.py    # Core analysis functions (normalization, on/off detection, co-contraction)
├── subject.py               # Subject and Trial data management classes
├── containers.py            # Data container classes (MarkerData, EMGData, ForceData, etc.)
├── subject_cache.py         # Data loading and caching script
├── utils.py                 # Utility functions (file I/O, pickle operations)
├── visualizations.py        # Visualization and plotting scripts
├── signalprocessing/        # Signal processing module
│   ├── filters.py           # Butterworth filter implementations
│   ├── arithmetics.py       # Signal arithmetic operations
│   ├── general.py           # General signal operations
│   ├── math_functions.py    # Mathematical functions
│   ├── spectral_analysis.py # FFT and spectral analysis
│   ├── stats.py             # Statistical functions
│   └── ...
├── successful_trials.json   # List of successful trial names
├── early_trials.json        # List of early latency trials
└── requirements.txt         # Python dependencies
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Talhauzumcu/Gaitinitiation_EMG_analysis.git
cd Gaitinitiation_EMG_analysis
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn

---

## Usage

### 1. Prepare Subject Cache
First, run `subject_cache.py` to load QTM data and create a cached pickle file:

```bash
python subject_cache.py
```

This script:
- Loads motion capture data from QTM MAT files
- Extracts EMG channels (channels 33, 34, 35, 38, 39, 40)
- Loads event timing data
- Marks trials as successful/unsuccessful and early/late
- Saves processed subjects to `subjects_cache.pkl`

### 2. Run Analysis
Execute the main analysis script:

```bash
python analysis.py
```

### 3. Generate Visualizations
```bash
python visualizations.py
```

---

## Data Pipeline

```
QTM MAT Files → Subject Loading → EMG Extraction → Preprocessing → Analysis → CSV/Plots
```

1. **Data Loading** (`subject.py`, `subject_cache.py`):
   - Load QTM-exported MAT files containing marker trajectories, analog signals, and force plate data
   - Extract EMG channels from analog data
   - Load event timing data (green cue, CoP onset, stop signal, etc.)

2. **Preprocessing** (`containers.py`, `analysis_functions.py`):
   - EMG signal processing (see below)

3. **Analysis** (`analysis.py`):
   - Compute metrics across defined time windows
   - Generate results as CSV files

---

## EMG Signal Preprocessing

The EMG preprocessing pipeline is implemented in the `EMGData` class (`containers.py`) and `find_emg_on_off()` function (`analysis_functions.py`):

### Processing Steps

```python
# 1. Remove NaN values
emg_signal[np.isnan(emg_signal)] = 0

# 2. Full-wave rectification (squaring)
rectified_signal = np.power(emg_signal, 2)

# 3. Low-pass filtering (linear envelope)
# Butterworth filter: 6 Hz cutoff, 2nd order, zero-phase (filtfilt)
smoothed_signal = sp.filters.low_pass(rectified_signal, cutoff_freq=6, sampling_rate=2000, order=2)

# 4. Amplitude normalization (0-1 scale)
normalized_signal = emg_signal / np.max(emg_signal)
```

### Technical Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling Rate | 2000 Hz | Analog data acquisition rate |
| Rectification | Full-wave (x²) | Squared signal values |
| Envelope Filter | 6 Hz low-pass | Butterworth, 2nd order |
| Filter Type | Zero-phase | `scipy.signal.filtfilt` |
| Normalization | Peak amplitude | Scaled to 0-1 range |

---

## Analyses

### EMG Amplitude Analysis

Computed metrics for each time window:

1. **Integrated EMG (iEMG)**: Area under the envelope curve
   ```python
   iEMG = np.trapezoid(envelope_signal[start:end])
   ```

2. **Mean Amplitude**: Average envelope amplitude
   ```python
   amplitude_mean = np.mean(envelope_signal[start:end])
   ```

3. **Average Peaks**: Mean of 10 highest amplitude values
   ```python
   average_peaks = np.mean(sorted(signal[start:end], reverse=True)[:10])
   ```

### EMG On/Off Detection

Binary muscle activation detection (`find_emg_on_off()`):

#### Algorithm
1. Process EMG signal (rectify, smooth, normalize)
2. Calculate baseline from 2-second window before green cue
3. Set threshold at baseline + 3% of max amplitude
4. Apply threshold to create binary on/off signal
5. Remove short activations (< 50 ms minimum duration)
6. Merge nearby activations (< 200 ms gap)

```python
baseline = np.mean(signal[green - 2*fs : green])
threshold = baseline + 0.03  # 3% above baseline
on_off_signal = (signal > threshold).astype(int)
```

#### Post-processing
- **Minimum duration**: 50 ms - activations shorter than this are removed
- **Gap merging**: 200 ms - activations separated by less than this are merged

### Muscle Co-contraction Analysis

Quantifies simultaneous activation of antagonist muscle pairs using Winter's method:

```python
# Calculate overlapping activation area
common_activations = np.minimum(normalized_emg1, normalized_emg2)
common_area = np.trapezoid(common_activations)
total_area = np.trapezoid(emg1) + np.trapezoid(emg2)

cocontraction_percentage = (2 * common_area / total_area) * 100
```

#### Muscle Pairs Analyzed
| Agonist | Antagonist |
|---------|------------|
| Right Tibialis Anterior | Right Soleus |
| Left Tibialis Anterior | Left Soleus |
| Right Tibialis Anterior | Right Gastrocnemius Medialis |
| Left Tibialis Anterior | Left Gastrocnemius Medialis |

---

## Key Events and Time Windows

### Event Markers

| Event | Description |
|-------|-------------|
| `green` | Green "GO" cue presentation |
| `CoP_onset` | Center of Pressure movement onset |
| `stopsignal` | Stop signal presentation (+27ms monitor delay) |
| `post_peak` | Posterior CoP peak |
| `frontal_peak` | Anterior CoP peak |

### Analysis Time Windows

| Window | Start | End |
|--------|-------|-----|
| Pre-Green to CoP Onset | Green - 0.3s | CoP Onset |
| CoP Onset to Stop Perceived | CoP Onset | Stop Signal + 27ms |
| Stop Perceived to Post Peak | Stop Signal + 27ms | Posterior Peak |
| CoP Onset to Post Peak | CoP Onset | Posterior Peak |
| Post Peak to Frontal Peak | Posterior Peak | Frontal Peak |

---

## Output Files

### Analysis Results

| File | Description |
|------|-------------|
| `emg_analysis_results.csv` | iEMG, amplitude, and peak values per trial/muscle |
| `emg_cocontraction_results.csv` | Co-contraction percentages for muscle pairs |
| `emg_on_off_signals_START_pre_green_END_frontalpeak.csv` | Resampled (100 points) on/off signals |
| `emg_on_off_green_absolute.csv` | Absolute on/off timing relative to events |

### Columns in Main Results File
- `trial_name`: Trial identifier
- `subject_id`: Subject number
- `category`: YA (Young Adult) or OA (Older Adult)
- `success`: Trial outcome (True/False)
- `latency`: early or late condition
- `reaction_time(ms)`: Time from green cue to CoP onset
- `emg_channel`: Muscle name
- `iEMG(window)`: Integrated EMG for each time window
- `amplitude_mean(window)`: Mean amplitude for each window
- `average_peaks(window)`: Average of 10 highest peaks per window

---

## Visualizations

The `visualizations.py` script generates an extensive set of plots saved to the `./plots` directory. Data preprocessing includes outlier removal using z-score threshold (±3 SD) and reaction time filtering (0-1000 ms).

### 1. EMG Amplitude Comparisons (Boxplots with Jittered Points)

| Plot | Comparison | File |
|------|------------|------|
| YA Success vs Non-Success | Mean amplitude across 5 time windows | `emg_amplitude_YA_success_vs_non_success.png` |
| OA Success vs Non-Success | Mean amplitude across 5 time windows | `emg_amplitude_OA_success_vs_non_success.png` |
| OA vs YA | Age group comparison | `emg_amplitude_OA_vs_YA.png` |

### 2. Integrated EMG (iEMG) Comparisons (Violin Plots + Strip Plots)

| Plot | Comparison | File |
|------|------------|------|
| YA Success vs Non-Success | iEMG distribution | `iemg_YA_success_vs_non_success.png` |
| OA Success vs Non-Success | iEMG distribution | `iemg_OA_success_vs_non_success.png` |
| OA vs YA | Age group comparison | `iemg_OA_vs_YA.png` |

### 3. Average Peak Amplitude Comparisons (Bar Plots + Strip Plots)

| Plot | Comparison | File |
|------|------------|------|
| YA Success vs Non-Success | Top-10 peak average | `average_peaks_YA_success_vs_non_success.png` |
| OA Success vs Non-Success | Top-10 peak average | `average_peaks_OA_success_vs_non_success.png` |
| OA vs YA | Age group comparison | `average_peaks_OA_vs_YA.png` |

### 4. Muscle Co-contraction Comparisons

Two visualization types for each co-contraction pair across 3 time intervals:

**Boxplot versions:**
| Plot | File |
|------|------|
| YA Success vs Non-Success | `cocontraction_YA_success_vs_non_success.png` |
| OA Success vs Non-Success | `cocontraction_OA_success_vs_non_success.png` |
| OA vs YA | `cocontraction_OA_vs_YA.png` |

**Violin plot versions:**
| Plot | File |
|------|------|
| YA Success vs Non-Success | `cocontraction_violin_YA_success_vs_non_success.png` |
| OA Success vs Non-Success | `cocontraction_violin_OA_success_vs_non_success.png` |
| OA vs YA | `cocontraction_violin_OA_vs_YA.png` |

### 5. EMG Time Series Plots (Mean ± 95% CI)

Time-normalized EMG envelope from CoP onset to frontal peak (resampled to 200 points):

| Plot | Comparison | File |
|------|------------|------|
| YA vs OA | Age group comparison | `emg_timeseries_mean_CI_YA_vs_OA.png` |
| YA: Success vs Non-Success | Within young adults | `emg_timeseries_mean_CI_YA_success_vs_non_success.png` |
| OA: Success vs Non-Success | Within older adults | `emg_timeseries_mean_CI_OA_success_vs_non_success.png` |

### 6. EMG On/Off Signal Heatmaps

Binary activation patterns (grayscale heatmaps) showing proportion of time each muscle is "on":

| Plot | Description | File |
|------|-------------|------|
| YA vs OA (side-by-side) | Overall mean activation | `emg_on_off_signals_YA_vs_OA.png` |
| Early/Late × YA/OA | 2×2 grid by latency condition | `emg_on_off_signals_YA_vs_OA_wlatency.png` |
| YA vs OA (interleaved) | Direct comparison per muscle | `emg_on_off_signals_YA_vs_OA_interleaved.png` |
| Early/Late interleaved | Latency comparison | `emg_on_off_signals_YA_vs_OA_wlatency_interleaved.png` |

### 7. EMG On/Off Timing Boxplots

Horizontal boxplots showing onset/offset timing relative to green cue, with reference lines for mean CoP onset and frontal peak:

| Plot | Comparison | File |
|------|------------|------|
| YA vs OA | Age group timing differences | `emg_on_off_boxplots_YA_vs_OA.png` |
| Success vs Failure (by group) | Within YA and OA separately | `emg_on_off_boxplots_success_vs_failure_by_group.png` |

### Visualization Summary

| Category | Number of Plots |
|----------|-----------------|
| Amplitude comparisons | 3 |
| iEMG comparisons | 3 |
| Peak amplitude comparisons | 3 |
| Co-contraction (boxplot) | 3 |
| Co-contraction (violin) | 3 |
| Time series with CI | 3 |
| On/Off heatmaps | 4 |
| On/Off timing boxplots | 2 |
| **Total** | **24** |

---

## Signal Processing Module

The `signalprocessing` package provides:

### Filters (`filters.py`)
- `low_pass()`: Butterworth low-pass filter
- `high_pass()`: Butterworth high-pass filter
- `bandpass()`: Butterworth band-pass filter
- `band_block()`: Butterworth band-stop (notch) filter

All filters use zero-phase filtering (`scipy.signal.filtfilt`) to avoid phase distortion.

---

## References

- Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*. Wiley. (Co-contraction method, p. 152)
" 
