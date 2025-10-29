import os
import json
import scipy.io
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