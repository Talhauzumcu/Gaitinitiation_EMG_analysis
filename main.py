#%%
import signalprocessing as sp
import numpy as np
import scipy.io
import os
from subject import Subject, TrialData
import matplotlib.pyplot as plt

# %%
qtm_path = "D:\Freelance_data\Gerontology_dshs\gaitinitiation\qtm_output"
subjects_no_list = os.listdir(qtm_path)
subjects_no_list = ['02']
# %%
for subject_no in subjects_no_list:
    subject = Subject(subject_id=subject_no)
    trial_files = [f for f in os.listdir(os.path.join(qtm_path, subject_no)) if f.endswith('.mat')]    

    for trial_file in trial_files:
        # Load a MAT file
        filepath = os.path.join(qtm_path, subject_no, trial_file)

# %%
# Now load the trial
trial = subject.load_mat_file(filepath)

# %%
