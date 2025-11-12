import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import signalprocessing as sp
import csv


def normalize_emg(emg_signal):
    max_amplitude = np.max(emg_signal)
    normalized_signal = emg_signal / max_amplitude if max_amplitude != 0 else emg_signal

    return normalized_signal


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

def find_frontal_peak(trial):
    cop_data = sp.filters.low_pass(trial.forces['Bertec'].cop[:, 0], cutoff_freq=15, sampling_rate=2000, order=2)
    cop_offset = np.mean(cop_data[:50])
    cop_data = cop_data - cop_offset 
    posterior_peak = trial.events['post_peak']
    max_idx = np.argmax(cop_data[posterior_peak:posterior_peak + 1500]) + posterior_peak
    return max_idx

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

#%%
def get_cocontraction(emg1, emg2):
    """Takes raw emg signals and calculates cocontraction percentage. 
    Based on Winter p. 152"""

    #First process both emg signals
    emg1[np.isnan(emg1)] = 0  # Remove NaNs
    rectified_signal1 = np.power(emg1, 2)
    smoothed_signal1 = sp.filters.low_pass(rectified_signal1, cutoff_freq=6, sampling_rate=2000, order=2)
    normalized_signal1 = normalize_emg(smoothed_signal1)

    emg2[np.isnan(emg2)] = 0  # Remove NaNs
    rectified_signal2 = np.power(emg2, 2)
    smoothed_signal2 = sp.filters.low_pass(rectified_signal2, cutoff_freq=6, sampling_rate=2000, order=2)
    normalized_signal2 = normalize_emg(smoothed_signal2)

    #Calculate cocontraction
    common_activations = np.minimum(normalized_signal1, normalized_signal2)
    common_area = np.trapezoid(common_activations)
    total_area = np.trapezoid(normalized_signal1) + np.trapezoid(normalized_signal2)

    cocontraction = ((2 * common_area) / total_area) * 100 if total_area != 0 else 0

    return cocontraction


# %%
def plot_emg_cocontraction(trial, emg1, emg2, slice_idxs, plot_start_time, plot_end_time, fs, cocontraction_pct):
    
    emg1_data = emg1.get_processed_data()
    emg2_data = emg2.get_processed_data()
    emg1_name = emg1.name
    emg2_name = emg2.name

    emg1_section = emg1_data[slice_idxs[0]:slice_idxs[1]]
    emg2_section = emg2_data[slice_idxs[0]:slice_idxs[1]]
    overlap = np.minimum(emg1_section, emg2_section)

    # Create time vector for the segment
    time_vector = (np.arange(len(emg1_data)) / fs)
    plot_slice_start = int(plot_start_time * fs)
    plot_slice_end = int(plot_end_time * fs)
    time_vector = time_vector[plot_slice_start:plot_slice_end]
    emg1_plot = emg1_data[plot_slice_start:plot_slice_end]
    emg2_plot = emg2_data[plot_slice_start:plot_slice_end]
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both EMG signals
    ax.plot(time_vector, emg1_plot, label=emg1_name, color='blue', linewidth=0.5)
    ax.plot(time_vector, emg2_plot, label=emg2_name, color='red', linewidth=0.5)

    # Fill the overlap area
    ax.fill_between(time_vector[slice_idxs[0]-plot_slice_start:slice_idxs[1]-plot_slice_start], 0, overlap, color='grey', alpha=0.4, label='Cocontraction Area')

    # Add event markers (adjusted to segment time)
    green_event = trial.events['green']
    cop_onset_event = trial.events['CoP_onset']
    post_peak_event = trial.events['post_peak']
    frontal_peak_event = trial.events['frontal_peak']

    ax.axvline(green_event / fs, color='green', linestyle='--', alpha=0.7, label='Green Cue')
    ax.axvline(cop_onset_event / fs, color='blue', linestyle='--', alpha=0.7, label='CoP Onset')
    ax.axvline(post_peak_event / fs, color='orange', linestyle='--', alpha=0.7, label='Post Peak')
    ax.axvline(frontal_peak_event / fs, color='brown', linestyle='--', alpha=0.7, label='Frontal Peak')

    # Add cocontraction percentage text
    ax.text(0.98, 0.95, f'Cocontraction: {cocontraction_pct:.1f}%', 
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized EMG Amplitude', fontsize=12)
    ax.set_title(f'EMG Cocontraction - Trial {trial.trial_name}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(np.max(emg1_plot), np.max(emg2_plot)) * 1.1])

    plt.tight_layout()
    plt.show()
# %%
