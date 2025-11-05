#%%
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent figures from opening
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

#%% Add the frontal peak event to each trial
def find_frontal_peak(trial):
    cop_data = sp.filters.low_pass(trial.forces['Bertec'].cop[:, 0], cutoff_freq=15, sampling_rate=2000, order=2)
    cop_offset = np.mean(cop_data[:50])
    cop_data = cop_data - cop_offset 
    posterior_peak = trial.events['post_peak']
    max_idx = np.argmax(cop_data[posterior_peak:posterior_peak + 1500]) + posterior_peak
    return max_idx

for subject in subjects:
    for name, trial in subject.trials.items():
        frontal_peak = find_frontal_peak(trial)
        subject.trials[name].events['frontal_peak'] = frontal_peak

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
                        'copOnset_postPeak': np.trapezoid(envelope_signal[trial.events['CoP_onset']:trial.events['post_peak']]),
                        'postPeak_frontPeak': np.trapezoid(envelope_signal[trial.events['post_peak']:trial.events['frontal_peak']])}
                
                amplitude_mean = {'preGreen_copOnset': np.mean(envelope_signal[trial.events['green'] - int(pre_green * analog_fs):trial.events['CoP_onset']]),
                        'copOnset_stopPerceived': np.mean(envelope_signal[trial.events['start']:stop_perceived]),
                        'stopPerceived_postPeak': np.mean(envelope_signal[stop_perceived:trial.events['post_peak']]),
                        'copOnset_postPeak': np.mean(envelope_signal[trial.events['CoP_onset']:trial.events['post_peak']]),
                        'postPeak_frontPeak': np.mean(envelope_signal[trial.events['post_peak']:trial.events['frontal_peak']])
                        }
                
                #First 10 peaks average (highest 10 values)
                average_peaks = {'preGreen_copOnset': np.mean(sorted(envelope_signal[trial.events['green'] - int(pre_green * analog_fs):trial.events['CoP_onset']], reverse=True)[:10]),
                                'copOnset_stopPerceived': np.mean(sorted(envelope_signal[trial.events['start']:stop_perceived], reverse=True)[:10]),
                        'stopPerceived_postPeak': np.mean(sorted(envelope_signal[stop_perceived:trial.events['post_peak']], reverse=True)[:10]),
                        'copOnset_postPeak': np.mean(sorted(envelope_signal[trial.events['CoP_onset']:trial.events['post_peak']], reverse=True)[:10]),
                        'postPeak_frontPeak': np.mean(sorted(envelope_signal[trial.events['post_peak']:trial.events['frontal_peak']], reverse=True)[:10])
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

#%%
# Save all results to CSV
save_analysis_results_to_csv(all_results, "emg_analysis_results.csv")
            
            
                

#%%
def normalize_emg(emg_signal):
    max_amplitude = np.max(emg_signal)
    normalized_signal = emg_signal / max_amplitude if max_amplitude != 0 else emg_signal

    return normalized_signal

# %%
def find_emg_on_off(emg_signal, sampling_rate, events):
    try:
        # Rectify and smooth the EMG signal
        emg_signal = emg_signal[~np.isnan(emg_signal)]  # Remove NaNs
        rectified_signal = np.power(emg_signal, 2)
        smoothed_signal = sp.filters.low_pass(rectified_signal, cutoff_freq=6, sampling_rate=sampling_rate, order=2)
        normalized_signal = normalize_emg(smoothed_signal)
        green_idx = events['green']
        baseline_window = 2 * sampling_rate  # 2 second window to use before green as a baseline
        baseline = np.mean(normalized_signal[green_idx - baseline_window:green_idx])  # Use the window before green for baseline
        threshold = baseline + 0.03 # 3% of max amplitude above baseline
        above_threshold = np.where(normalized_signal > threshold)
        on_off_signal = np.zeros_like(normalized_signal)
        on_off_signal[above_threshold] = 1

        minimum_duration = int(0.05 * sampling_rate)  # Minimum duration of 50 ms
        # if any interval is shorter than minimum_duration, set it to 0
        current_start = None
        for i in range(len(on_off_signal)):
            if on_off_signal[i] == 1 and current_start is None:
                current_start = i
            elif on_off_signal[i] == 0 and current_start is not None:
                duration = i - current_start
                if duration < minimum_duration:
                    on_off_signal[current_start:i] = 0
                current_start = None

        # if two intervals are seperated only by a short gap, merge them
        gap_threshold = int(0.2 * sampling_rate)  # 200 ms gap
        last_on = None
        for i in range(len(on_off_signal)):
            if on_off_signal[i] == 1:
                if last_on is not None and i - last_on < gap_threshold:
                    on_off_signal[last_on:i] = 1
                last_on = i
    except Exception as e:
        print(f"Error processing EMG signal: {e}")
        on_off_signal = np.zeros_like(emg_signal)

    return on_off_signal


# %% Calculate on off signals for the emgs and store them in the emg dataclass
analog_fs = 2000
for subject in subjects:
    for name, trial in subject.trials.items():
        for emg in trial.emgs.values():
            emg_signal = emg.get_data()
            on_off_signal = find_emg_on_off(emg_signal, analog_fs, trial.events)
            emg.on_off_signal = on_off_signal
#%%Plot emg On off signals for verification
for subject in subjects:
    for name, trial in subject.trials.items():
        for emg in trial.emgs.values():
            try:
                plt.figure(figsize=(12, 4))
                emg_data = emg.get_data()
                emg_data = emg_data[~np.isnan(emg_data)]  # Remove NaNs for plotting
                emg_data = np.power(emg_data, 2)  # Rectify
                emg_data = sp.filters.low_pass(emg_data, cutoff_freq=6, sampling_rate=analog_fs, order=2)  # Smooth
                emg_data = normalize_emg(emg_data)  # Normalize
                plt.plot(emg_data, label='Processed EMG')
                plt.plot(emg.on_off_signal * np.max(emg_data), label='On/Off Signal', alpha=0.7)
                plt.axvline(trial.events['green'], linestyle='--', label='Green Cue', color='green')
                plt.title(f'Subject {subject.subject_id} - Trial {trial.trial_name} - EMG {emg.name}')
                plt.xlabel('Time (s)')
                plt.ylabel('EMG Amplitude')

                #Save
                output_dir = Path('emg_on_off_plots')
                output_dir.mkdir(exist_ok=True)
                filename = f"subject_{subject.subject_id}_trial_{trial.trial_name}_emg_{emg.name}_on_off.png"
                plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error plotting {subject.subject_id} - {trial.trial_name} - {emg.name}: {e}")
                continue
# %% Create muscle activation timing plots for each trial
def plot_muscle_activation_timing(trial, subject_id, output_dir='muscle_activation_plots', sampling_rate=2000):

    emg_list = list(trial.emgs.values())
    if not emg_list:
        print(f"No EMG data found for trial {trial.trial_name}")
        return
    
    slice_start = trial.events['green'] - int(.2 * sampling_rate)  # .2 second before green
    slice_end = trial.events['frontal_peak'] + int(.2 * sampling_rate)  # .2 second after frontal peak
    cycle_length = slice_end - slice_start
        
    green = trial.events['green'] - slice_start  # Adjust green event to sliced data
    cop_onset = trial.events['CoP_onset'] - slice_start  # Adjust CoP_onset event to sliced data
    post_peak = trial.events['post_peak'] - slice_start  # Adjust post_peak event to sliced data
    stopsignal = trial.events['stopsignal'] - slice_start  # Adjust stop signal event to sliced data

    green_pct = (green  / cycle_length) * 100
    cop_onset_pct = (cop_onset / cycle_length) * 100
    post_peak_pct = (post_peak / cycle_length) * 100
    stopsignal_pct = (stopsignal / cycle_length) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, len(emg_list) * 0.5 + 1))
    
    ax.axvline(green_pct, color='green', linestyle='--', label='Green Cue')
    ax.axvline(cop_onset_pct, color='blue', linestyle='--', label='CoP Onset')
    ax.axvline(stopsignal_pct, color='red', linestyle='--', label='Stop Signal')
    ax.axvline(post_peak_pct, color='orange', linestyle='--', label='Post Peak')
    # Process each muscle
    muscle_names = []
    for idx, emg in enumerate(emg_list):
        muscle_names.append(emg.name)
        
        # Get the on/off signal for this muscle within the gait cycle
        on_off = emg.on_off_signal[slice_start:slice_end]

        # Find continuous activation periods
        activation_starts = []
        activation_ends = []
        in_activation = False
        
        for i, val in enumerate(on_off):
            if val == 1 and not in_activation:
                activation_starts.append(i)
                in_activation = True
            elif val == 0 and in_activation:
                activation_ends.append(i)
                in_activation = False
        
        if in_activation:
            activation_ends.append(len(on_off))
        
        for start, end in zip(activation_starts, activation_ends):
            start_pct = (start / cycle_length) * 100
            duration_pct = ((end - start) / cycle_length) * 100
            
            ax.barh(idx, duration_pct, left=start_pct, height=0.6, 
                color='steelblue', edgecolor='darkblue', linewidth=0.5)
    
    ax.set_yticks(range(len(muscle_names)))
    ax.set_yticklabels(muscle_names)
    ax.set_xlabel('Green - .2s to Frontal Peak + .2s', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(muscle_names) - 0.5)
    ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left')
    success_str = "Success" if trial.success else "Fail"
    latency_str = "Early" if trial.early else "Late"
    ax.set_title(f'Subject {subject_id} - {trial.trial_name}\n{success_str} - {latency_str}', 
                fontsize=10, pad=10)
    
    plt.tight_layout()
    
    # Save figure
    print(output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    filename = f"subject_{subject_id}_trial_{trial.trial_name}_activation.png"
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
 
    return str(output_path / filename)

# %% Generate plots for all trials
plot_count = 0
for subject in subjects:
    for trial_name, trial in subject.trials.items():
        try:
            plot_path = plot_muscle_activation_timing(trial, subject.subject_id, sampling_rate=analog_fs)
            if plot_path:
                plot_count += 1
                if plot_count % 10 == 0:
                    print(f"Generated {plot_count} plots...")
        except Exception as e:
            print(f"Error plotting {subject.subject_id} - {trial_name}: {e}")

print(f"Complete! Generated {plot_count} muscle activation timing plots.")
print(f"Plots saved in 'muscle_activation_plots' directory.")


# %%
