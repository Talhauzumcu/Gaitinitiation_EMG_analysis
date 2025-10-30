#%%
import matplotlib.pyplot as plt
import signalprocessing as sp
import numpy as np
from utils import *
from scipy.signal import find_peaks
import csv
import os
from pathlib import Path

#%%
def save_analysis_results_to_csv(all_results, filename="analysis_results.csv"):
    """
    Save all analysis results to a CSV file.
    
    Parameters:
    -----------
    all_results : list of dict
        List of dictionaries containing analysis results for each trial/emg combination
    filename : str
        Output CSV filename (default: "analysis_results.csv")
    """
    if not all_results:
        print("No results to save!")
        return
    
    # Get all unique column names from all results
    all_keys = set()
    for result in all_results:
        all_keys.update(result.keys())
    
    # Define the order of base columns
    base_columns = ['trial_name', 'subject_id', 'category', 'success', 'latency', 'emg_channel']
    
    # Get all analysis result columns (sorted for consistency)
    analysis_columns = sorted([k for k in all_keys if k not in base_columns])
    
    # Final column order
    fieldnames = base_columns + analysis_columns
    
    # Write to CSV
    csv_path = Path(filename)
    print(f"Saving {len(all_results)} rows to {csv_path}...")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Results successfully saved to {csv_path}")
    print(f"Columns: {', '.join(fieldnames)}")

#%%
#Load subject data from cache. If the cache is not yet prepared, run the subject_cache.py script first.
subjects = load_subjects_pickle("subjects_cache.pkl")
# %% Start EMG analysis 
marker_fs = 200
analog_fs = 2000

# List to store all results
all_results = []

for subject in subjects:
    # Determine category (OA or YA)
    category = 'YA' if subject.is_young else 'OA'
    
    for trial in subject.trials.values():
        for emg in trial.emgs.values():
            try:
                raw_signal = emg.get_data()
                raw_signal[np.isnan(raw_signal)] = 0  # Replace NaNs with zeros
                time_vector = np.arange(len(raw_signal)) / analog_fs

                # Filtering
                # filtered_signal = sp.filters.bandpass(raw_signal, low_freq=30, high_freq=300, sampling_rate=analog_fs, order=2)
                rectified_signal = np.abs(raw_signal)
                envelope_signal = sp.filters.low_pass(rectified_signal, cutoff_freq=6, sampling_rate=analog_fs, order=2)
                max_amplitude = np.max(envelope_signal[trial.events['green']:]) # Consider data after 'green' event
                envelope_signal = envelope_signal / max_amplitude  # Normalize to max amplitude after 'green' event

                pre_green = 0.3 # seconds before green cue
                stop_perceived = int(trial.events['stopsignal'] + 0.027 * analog_fs) # 54 ms delay before monitor shows stop signal
                iEMG = {'preGreen_copOnset': np.trapezoid(envelope_signal[trial.events['green'] - int(pre_green * analog_fs):trial.events['CoP_onset']]),
                        'copOnset_stopPerceived': np.trapezoid(envelope_signal[trial.events['start']:stop_perceived]),
                        'stopPerceived_postPeak': np.trapezoid(envelope_signal[stop_perceived:trial.events['post_peak']]),
                        'copOnset_postPeak': np.trapezoid(envelope_signal[trial.events['CoP_onset']:trial.events['post_peak']])}
                
                amplitude_mean = {'preGreen_copOnset': np.mean(envelope_signal[trial.events['green'] - int(pre_green * analog_fs):trial.events['CoP_onset']]),
                        'copOnset_stopPerceived': np.mean(envelope_signal[trial.events['start']:stop_perceived]),
                        'stopPerceived_postPeak': np.mean(envelope_signal[stop_perceived:trial.events['post_peak']]),
                        'copOnset_postPeak': np.mean(envelope_signal[trial.events['CoP_onset']:trial.events['post_peak']])
                        }
                
                #First 10 peaks average (highest 10 values)
                average_peaks = {'preGreen_copOnset': np.mean(sorted(envelope_signal[trial.events['green'] - int(pre_green * analog_fs):trial.events['CoP_onset']], reverse=True)[:10]),
                                'copOnset_stopPerceived': np.mean(sorted(envelope_signal[trial.events['start']:stop_perceived], reverse=True)[:10]),
                        'stopPerceived_postPeak': np.mean(sorted(envelope_signal[stop_perceived:trial.events['post_peak']], reverse=True)[:10]),
                        'copOnset_postPeak': np.mean(sorted(envelope_signal[trial.events['CoP_onset']:trial.events['post_peak']], reverse=True)[:10])
                        }
                
                # Create result dictionary for this EMG channel
                result = {
                    'trial_name': trial.trial_name,
                    'subject_id': subject.subject_id,
                    'category': category,
                    'success': trial.success,
                    'latency': 'early' if trial.early else 'late',
                    'emg_channel': emg.name,
                }
                
                # Add iEMG results with prefix
                for key, value in iEMG.items():
                    result[f'iEMG({key})'] = value
                
                # Add amplitude_mean results with prefix
                for key, value in amplitude_mean.items():
                    result[f'amplitude_mean({key})'] = value

                # Add average_peaks results with prefix
                for key, value in average_peaks.items():
                    result[f'average_peaks({key})'] = value

                all_results.append(result)
            except Exception as e:
                print(f"Error processing {subject.subject_id} {trial.trial_name} {emg.name}: {e}")
# Save all results to CSV
save_analysis_results_to_csv(all_results, "emg_analysis_results.csv")
            
            
                

#%%
def find_frontal_peak(trial):
    BERTEC_ORIGIN = -307.92 #Bertecs origin in the lab frame. Some of the data is in local frame and some in lab frame.
    cop_data = sp.filters.low_pass(trial.forces['Bertec'].cop[:, 0], cutoff_freq=15, sampling_rate=2000, order=2)
    cop_offset = np.mean(cop_data[:50])
    cop_data = cop_data + BERTEC_ORIGIN - cop_offset #Align COP data to lab frame
    posterior_peak = trial.events['post_peak']
    peaks, _ = find_peaks(cop_data[posterior_peak:], distance=500, height=-300, prominence=50)
    if len(peaks) > 1:
        print(f"WARNING: Multiple frontal peaks found Using the first one.")
    if len(peaks) == 0:
        print(f"WARNING: No frontal peak found")
        return 0, cop_data
    return peaks[0] + posterior_peak, cop_data

    peak = peaks[0] + posterior_peak
# %%
subject = subjects[4]
for name, trial in subject.trials.items():

    frontal_peak, cop_data = find_frontal_peak(trial)
    fig, ax = plt.subplots()
    ax.plot(cop_data, label='CoP AP')
    ax.plot(frontal_peak, cop_data[frontal_peak], 'ro', label='Frontal Peak')
    
    
# %%
