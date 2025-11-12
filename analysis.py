#%%
import matplotlib.pyplot as plt
import signalprocessing as sp
import numpy as np
from utils import *
from scipy.signal import find_peaks
import os
from pathlib import Path
from analysis_functions import * 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent figures from opening

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
# Save all results to CSV
save_analysis_results_to_csv(all_results, "emg_analysis_results.csv")
            
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


# %% EMG cocontraction analysis
output_dir = Path('emg_cocontraction_results')
output_dir.mkdir(exist_ok=True)
filename = f"emg_cocontraction_results.csv"
output_path = output_dir / filename

analog_fs = 2000
pre_green = 0.3
idx_buffer = int(pre_green * analog_fs)
cocontraction_pairs = [('03_ri_tib_ant', '01_ri_soleus'), 
                       ('06_le_tib_ant', '07_le_soleus'),
                       ('03_ri_tib_ant', '02_ri_gastroc_med'), 
                       ('06_le_tib_ant', '08_le_gastroc_med')]  
for subject in subjects:
    for name, trial in subject.trials.items():
        line = f"{subject.subject_id}, {trial.trial_name} "
        for pair in cocontraction_pairs:
            emg1_name, emg2_name = pair
            emg1 = trial.emgs.get(emg1_name)
            emg2 = trial.emgs.get(emg2_name)
            try: 
                start_idx = trial.events['green'] - idx_buffer
                end_idx = trial.events['frontal_peak']
                emg1_segment = emg1.data[start_idx:end_idx]
                emg2_segment = emg2.data[start_idx:end_idx]
                cocontraction_pct = get_cocontraction(emg1_segment, emg2_segment)
                line += f",{cocontraction_pct:.1f} "
                print(f"Subject {subject.subject_id}, Trial {trial.trial_name}, "
                      f"Interval green-frontal_peak, Cocontraction between {emg1_name} "
                      f"and {emg2_name}: {cocontraction_pct:.2f}%")

                #Create and save the plot
                plot_emg_cocontraction(trial,
                                        emg1, 
                                        emg2, 
                                        slice_idxs=(start_idx, end_idx), 
                                        plot_start_time=(start_idx/analog_fs - .5),
                                        plot_end_time=(end_idx/analog_fs + .5), 
                                        fs=analog_fs, 
                                        cocontraction_pct=cocontraction_pct)
                #Save the plot
                plot_filename = f"{subject.subject_id}_{trial.trial_name}_{emg1_name}_{emg2_name}_cocontraction.png"
                plot_path = output_dir / plot_filename
                plt.savefig(plot_path)
                plt.close()

            except Exception as e:
                line += ",nan "
                print(f"Error calculating cocontraction for {subject.subject_id} {trial.trial_name} between {emg1_name} and {emg2_name}: {e}")

    # Write to CSV
    
        write_header = not output_path.exists()
        with open(output_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if write_header:
                header_parts = ['Subject_ID', 'Trial_Name']
                for pair in cocontraction_pairs:
                    emg1_name, emg2_name = pair
                    header_parts.append(f'Cocontraction_{emg1_name}_{emg2_name}_green_frontalPeak')
                csvwriter.writerow(header_parts)
            line_parts = line.strip().split(',')
            csvwriter.writerow(line_parts)

# %%
