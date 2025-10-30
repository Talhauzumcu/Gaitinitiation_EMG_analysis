import os
import json
import scipy.io
import pickle
from pathlib import Path

def create_success_list(filepath):
    """Creates an array containing the successful trial names in a given directory."""
    success_list = []
    for file in os.listdir(filepath):
        if file.endswith('.mat'):
            trial_name = os.path.splitext(file)[0]
            success_list.append(trial_name)
    return success_list

def save_list(list, filename):
    """Saves the successful trial names to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(list, f)

def get_files(filepath):
    """
    Gets all mat files in a directory and its subdirectories.
    """
    mat_files = {}
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(".mat"):
                mat_files[file] = {'path': os.path.join(root, file), 'root': root, 'filename': file} 
    return mat_files

def create_early_list(filepath):
    """Creates an array containing the early trial names in a given directory."""
    early_list = []
    late_list = []
    mat_files_dict = get_files(filepath)

    for filename, value in mat_files_dict.items():
        if filename.endswith('.mat'):
            #standardize trial names so they are compatible with qtm trials
            trial_name = os.path.splitext(filename)[0]
            trial_number = trial_name.split('_')[-1]
            std_trial_name = trial_name.replace(f"NOGO_{trial_number}", f"GO00{trial_number}")
            mat_file = scipy.io.loadmat(value['path'])
            data = mat_file['data']
            subj_fieldname = data.dtype.names[0]
            subj_data = data[subj_fieldname][0,0]
            latency = int(subj_data['latency'][0,0][0])
            if latency == 10 or latency == 5:
                early_list.append(std_trial_name)
    return early_list

def save_subjects_pickle(subjects, filename="subjects_cache.pkl"):
    """
    Save subjects list to a pickle file for faster loading.
    
    Parameters:
    -----------
    subjects : list
        List of Subject objects to save
    filename : str
        Name of the pickle file (default: "subjects_cache.pkl")
    """
    pickle_path = Path(filename)
    print(f"Saving {len(subjects)} subjects to {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(subjects, f)
    print(f"Subjects successfully saved to {pickle_path}")
    print(f"File size: {pickle_path.stat().st_size / (1024*1024):.2f} MB")

def load_subjects_pickle(filename="subjects_cache.pkl"):
    """
    Load subjects list from a pickle file.
    
    Parameters:
    -----------
    filename : str
        Name of the pickle file to load (default: "subjects_cache.pkl")
    
    Returns:
    --------
    list
        List of Subject objects, or None if file doesn't exist
    """
    pickle_path = Path(filename)
    if not pickle_path.exists():
        print(f"Pickle file '{pickle_path}' not found. Subjects need to be loaded from source.")
        return None
    
    print(f"Loading subjects from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        subjects = pickle.load(f)
    print(f"Successfully loaded {len(subjects)} subjects")
    return subjects

if __name__ == "__main__":
    # root_path = r"C:\Users\talha\OneDrive\Belgeler\GitHub\MarginOfStability\margin_of_stability\results\successful"
    # OA_list_path = os.path.join(root_path, "OA")
    # YA_list_path = os.path.join(root_path, "YA")
    # OA_success_list = create_success_list(OA_list_path)
    # YA_success_list = create_success_list(YA_list_path)
    # all_success_list = OA_success_list + YA_success_list
    # save_list(all_success_list, 'successful_trials.json')

    psytoolb_path = r"D:\Freelance_data\Gerontology_dshs\gaitinitiation\psytoolb_output"
    early_list= create_early_list(psytoolb_path)
    save_list(early_list, 'early_trials.json')

    #%%