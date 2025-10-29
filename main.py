#%%
import signalprocessing as sp
import numpy as np
import os
from subject import Subject
import matplotlib.pyplot as plt

# %%
QTM_PATH = "D:\Freelance_data\Gerontology_dshs\gaitinitiation\qtm_output"
PSY_PATH = "D:\Freelance_data\Gerontology_dshs\gaitinitiation\psytoolb_output"
EVENTS_PATH = "D:\Freelance_data\Gerontology_dshs\gaitinitiation\events\events.mat"
SUCCESS_FILE = "successful_trials.json"
EARLY_FILE = "early_trials.json"
YOUNG_SUBJECTS = np.concatenate((np.arange(1,11,1), np.array([21])))
EMG_CHANNELS = np.array([33,34,35,38,39,40])
subjects_no_list = os.listdir(QTM_PATH)
subjects = []
# %%
for subject_no in subjects_no_list:
    subject = Subject(subject_id=subject_no)
    subject.is_young = int(subject_no) in YOUNG_SUBJECTS
    trial_files = [f for f in os.listdir(os.path.join(QTM_PATH, subject_no)) if f.endswith('.mat')]

    for trial_file in trial_files:
        qtm_filepath = os.path.join(QTM_PATH, subject_no, trial_file)
        subject.load_qtm_data(qtm_filepath)

    subject.load_event_data(EVENTS_PATH)
    subject.set_trial_success(SUCCESS_FILE)
    subject.set_trial_latency(EARLY_FILE)
    subject.set_EMG_data(EMG_CHANNELS)
    subjects.append(subject)
# %% Start EMG analysis 
marker_fs = 200
analog_fs = 2000
for subject in subjects:
    for trial in subject.trials.values():
        for emg in trial.emgs.values():
            try:
                raw_signal = emg.get_data()
                raw_signal[np.isnan(raw_signal)] = 0  # Replace NaNs with zeros
                time_vector = np.arange(len(raw_signal)) / analog_fs

                # Filtering
                filtered_signal = sp.filters.bandpass(raw_signal, low_freq=30, high_freq=300, sampling_rate=analog_fs, order=2)
                rectified_signal = np.abs(filtered_signal)
                envelope_signal = sp.filters.low_pass(rectified_signal, cutoff_freq=3, sampling_rate=analog_fs, order=2)

                # Plotting
                plt.figure(figsize=(12, 6))
                # plt.plot(time_vector[:len(downsampled_signal)], downsampled_signal, label='Downsampled EMG')
                # plt.plot(time_vector[:len(raw_signal)], raw_signal, label='Raw EMG')
                plt.plot(time_vector[:len(rectified_signal)], rectified_signal, label='Rectified EMG', alpha=0.5)
                plt.plot(time_vector[:len(envelope_signal)], envelope_signal, label='EMG Envelope')
                plt.plot(time_vector[trial.events['green']], envelope_signal[trial.events['green']], 'go', label='Go Cue')
                plt.plot(time_vector[trial.events['start']], envelope_signal[trial.events['start']], 'ro', label='Start Cue')
                plt.plot(time_vector[trial.events['stopsignal']], envelope_signal[trial.events['stopsignal']], 'ko', label='Stop Signal')
                plt.plot(time_vector[trial.events['CoP_onset']], envelope_signal[trial.events['CoP_onset']], 'mo', label='CoP Onset')
                plt.plot(time_vector[trial.events['post_peak']], envelope_signal[trial.events['post_peak']], 'co', label='Post Peak')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title(f'{subject.subject_id} {trial.trial_name} {trial.success} EMG Signal Envelope with events - {emg.name}')
                
                
                age_dir = 'YA' if subject.is_young else 'OA'
                save_path = f'outputs/{age_dir}/{subject.subject_id}/{trial.trial_name}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                plt.savefig(f'{save_path}/{emg.name}_with_events.png')
                plt.cla()
                plt.close('all')
            except Exception as e:
                plt.close('all')
                print(f"Error processing {subject.subject_id} {trial.trial_name} {emg.name}: {e}")


#%%

    
    
# %%
