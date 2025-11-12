'''
This script loads the necessary data files, 
processes them into Subject objects, 
and saves them into a pickle file for faster future loading.
'''
import numpy as np
import os
from subject import Subject
from utils import *
from pathlib import Path
from analysis_functions import find_frontal_peak

QTM_PATH = Path("/mnt/Data/Freelance_data/Gerontology_dshs/gaitinitiation/qtm_output")
PSY_PATH = Path("/mnt/Data/Freelance_data/Gerontology_dshs/gaitinitiation/psytoolb_output")
EVENTS_PATH = Path("/mnt/Data/Freelance_data/Gerontology_dshs/gaitinitiation/events/events.mat")
SUCCESS_FILE = "successful_trials.json"
EARLY_FILE = "early_trials.json"
PICKLE_FILE = "subjects_cache.pkl"  # Pickle cache file
YOUNG_SUBJECTS = np.concatenate((np.arange(1,11,1), np.array([21])))
EMG_CHANNELS = np.array([33,34,35,38,39,40])
subjects_no_list = os.listdir(QTM_PATH)

subjects = []
for subject_no in subjects_no_list:
    print(f"Processing subject {subject_no}...")
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

# Add the frontal peak event to the events if necessary
for subject in subjects:
    for name, trial in subject.trials.items():
        frontal_peak = find_frontal_peak(trial)
        subject.trials[name].events['frontal_peak'] = frontal_peak

# Save subjects to pickle for faster loading next time
save_subjects_pickle(subjects, PICKLE_FILE)




