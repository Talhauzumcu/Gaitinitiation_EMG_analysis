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
EMG_CHANNELS = np.array([33,34,35,38,39,40])
subjects_no_list = os.listdir(QTM_PATH)
subjects_no_list = ['02']
# %%
for subject_no in subjects_no_list:
    subject = Subject(subject_id=subject_no)
    trial_files = [f for f in os.listdir(os.path.join(QTM_PATH, subject_no)) if f.endswith('.mat')]

    for trial_file in trial_files:
        qtm_filepath = os.path.join(QTM_PATH, subject_no, trial_file)
        subject.load_qtm_data(qtm_filepath)

    subject.load_event_data(EVENTS_PATH)
    subject.set_trial_success(SUCCESS_FILE)
    subject.set_trial_latency(EARLY_FILE)
    subject.set_EMG_data(EMG_CHANNELS)

# %% Start EMG analysis 

marker_fs = 200
analog_fs = 2000
trial_1 = subject.get_trial_by_idx(0)

for key,val in trial_1.analogs.items():
    print(f"Analog Channel: {key}, Channel: {val.channel}, name: {val.name}, Sampling Rate: {val.sampling_rate}, Data shape: {val.data.shape}")
# %%
