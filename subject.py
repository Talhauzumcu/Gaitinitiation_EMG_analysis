import numpy as np
import scipy.io
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from containers import *
import os
import json

class Subject:
    """Class for managing subject data from QTM exports."""
    
    def __init__(self, subject_id: str = ""):
        self.subject_id = subject_id
        self.trials: Dict[str, TrialData] = {}
        
    def get_successful_trials(self) -> List[str]:
        """Get list of successful trial names."""
        return [name for name, trial in self.trials.items() if trial.success]
    
    def get_early_trials(self) -> List[str]:
        """Get list of early trial names."""
        return [name for name, trial in self.trials.items() if trial.early]
    
    def get_late_trials(self) -> List[str]:
        """Get list of late trial names."""
        return [name for name, trial in self.trials.items() if not trial.early]
    
    def get_unsuccessful_trials(self) -> List[str]:
        """Get list of unsuccessful trial names."""
        return [name for name, trial in self.trials.items() if not trial.success]
    
    def get_trial(self, trial_name: str) -> Optional[TrialData]:
        """Get trial data by name."""
        return self.trials.get(trial_name)
    
    def get_trial_names(self) -> List[str]:
        """Get list of all trial names."""
        return list(self.trials.keys())
    
    def print_trial_info(self, trial_name: str) -> None:
        """Print information about a trial."""
        trial = self.get_trial(trial_name)
        if trial is None:
            print(f"Trial '{trial_name}' not found.")
            return
    
        print(f"Trial: {trial.trial_name}")
        print(f"Markers ({len(trial.markers)}):", trial.get_marker_names()[:10], "..." if len(trial.markers) > 10 else "")
        print(f"Analogs ({len(trial.analogs)}):", trial.get_analog_names()[:10], "..." if len(trial.analogs) > 10 else "")
        print(f"Force Plates ({len(trial.forces)}):", trial.get_force_names())
        print(f"Metadata: {list(trial.metadata.keys())}")
        
    def get_trial_by_idx(self, index: int) -> Optional[TrialData]:
        """Get trial data by index."""
        trial_names = self.get_trial_names()
        if index < 0 or index >= len(trial_names):
            return None
        trial_name = trial_names[index]
        return self.trials[trial_name]
    
    def set_trial_success(self, success_file: str) -> None:
        """Set trial success flags based on provided success list."""
        with open(success_file, 'r') as f:
            success_list = json.load(f)
        for trial_name, trial in self.trials.items():
            trial.success = trial_name in success_list

    def set_trial_latency(self, early_file: str) -> None:
        """Set trial latency based on provided latency data."""
        with open(early_file, 'r') as f:
            early_data = json.load(f)
        for trial_name, trial in self.trials.items():
            #There are inconsistencies in trial naming in the early trials file
            trial.early = trial_name in early_data or trial_name.replace('0_', '') in early_data or \
                            trial_name.replace('1_', '') in early_data

    def load_qtm_data(self, filepath: str) -> TrialData:
        """
        Load QTM MAT file and parse into structured format.
        
        Args:
            filepath (str): Path to the .mat file
            
        Returns:
            TrialData: Parsed trial data
        """
        # Load the MAT file
        mat_data = scipy.io.loadmat(filepath)
        
        # Extract trial name from filepath or mat structure
        trial_name = self._extract_trial_name(filepath, mat_data)
        
        # Create trial data object
        trial = TrialData(trial_name=trial_name)
        
        # Parse the MAT file structure
        trial = self._parse_mat_structure(mat_data, trial_name, trial)
        
        # Store trial
        self.trials[trial_name] = trial
        
        return trial
    
    def load_event_data(self, filepath: str) -> None:
        mat_data = scipy.io.loadmat(filepath)
        events = mat_data['events']
        for trial in self.trials.values():
            subject_field = f"Subj_{self.subject_id}"
            trial_name = f"qtm_{trial.trial_name}"
            trial_event_data = events[subject_field][0,0][trial_name][0,0]
            event_names = trial_event_data.dtype.names
            for event_name in event_names:
                try:
                    if event_name == 'stopsignal':
                        trial.events[event_name] = trial_event_data[event_name][0,0][0,1]
                    else:
                        trial.events[event_name] = trial_event_data[event_name][0,0][0,0]
                except Exception as e:
                    print(f"Warning: Could not load event '{event_name}' for subject '{self.subject_id}' for trial '{trial_name}': {e}")
            
    def _extract_trial_name(self, filepath: str, mat_data: Dict) -> str:
        """Extract trial name from filepath or mat data."""
       
        # Try to get from filename
        filename = os.path.basename(filepath)
        trial_name = filename.replace('.mat', '')
        
        # Check if trial name exists as key in mat_data
        if trial_name in mat_data:
            return trial_name
        
        # Otherwise find the main data key (skip metadata keys)
        metadata_keys = ['__header__', '__version__', '__globals__']
        data_keys = [k for k in mat_data.keys() if k not in metadata_keys]
        
        if data_keys:
            return data_keys[0]
        
        return trial_name
    
    def _parse_mat_structure(self, mat_data: Dict, trial_name: str, trial: TrialData) -> TrialData:
        """
        Parse MAT file structure into organized trial data.
        
        QTM typically exports data in a structured array format.
        """
        # Get the main data structure
        if trial_name in mat_data:
            main_data = mat_data[trial_name]
        else:
            # Find first non-metadata key
            metadata_keys = ['__header__', '__version__', '__globals__']
            data_keys = [k for k in mat_data.keys() if k not in metadata_keys]
            if not data_keys:
                raise ValueError("No data found in MAT file")
            main_data = mat_data[data_keys[0]]
        
        # Handle different QTM export formats
        if isinstance(main_data, np.ndarray):
            trial = self._parse_structured_array(main_data, trial)
        elif isinstance(main_data, dict):
            trial = self._parse_dict_structure(main_data, trial)
        
        return trial
    
    def _parse_structured_array(self, data: np.ndarray, trial: TrialData) -> TrialData:
        """Parse MATLAB structured array format."""
        # MATLAB structures are often stored as numpy structured arrays
        if data.dtype.names:
            # This is a structured array with field names
            for field_name in data.dtype.names:
                field_data = data[field_name][0, 0]
                self._process_field(field_name, field_data, trial)
        else:
            # This might be a cell array or object array
            # Flatten if needed
            data = data.flatten()
            if len(data) > 0 and hasattr(data[0], 'dtype'):
                if data[0].dtype.names:
                    for field_name in data[0].dtype.names:
                        field_data = data[0][field_name]
                        self._process_field(field_name, field_data, trial)
        
        return trial
    
    def _parse_dict_structure(self, data: Dict, trial: TrialData) -> TrialData:
        """Parse dictionary structure."""
        for key, value in data.items():
            self._process_field(key, value, trial)
        return trial
    
    def _process_field(self, field_name: str, field_data: Any, trial: TrialData) -> None:
        """Process individual field from MAT structure."""
        field_name_lower = field_name.lower()
        
        # Try to identify if this is marker data, analog data, or metadata
        if 'trajectories' in field_name_lower:
            self._extract_qtm_trajectories(field_data, trial)
        elif 'analog' in field_name_lower:
            self._extract_qtm_analogs(field_data, trial)
        elif 'force' in field_name_lower:
            self._extract_qtm_force(field_data, trial)
        elif 'framerate' in field_name_lower:
            if isinstance(field_data, np.ndarray) and field_data.size == 1:
                trial.metadata['FrameRate'] = float(field_data.flat[0])
        else:
            # Store as metadata
            if isinstance(field_data, np.ndarray) and field_data.size == 1:
                trial.metadata[field_name] = field_data.item()
            else:
                trial.metadata[field_name] = field_data
    
    def _extract_qtm_trajectories(self, traj_data: Any, trial: TrialData) -> None:
        """Extract marker trajectories from QTM Trajectories field."""
        if not isinstance(traj_data, np.ndarray) or traj_data.shape != (1, 1):
            return
        
        try:
            # Unwrap the (1,1) structure
            unwrapped = traj_data[0, 0]
            
            # unwrapped[0] contains 'Labeled' markers (structured array)
            # unwrapped[1] contains 'Unidentified' markers
            if len(unwrapped) > 0:
                labeled = unwrapped[0]
                
                # labeled is shape (1, 1) with dtype that has 'Labels' and 'Data' fields
                if labeled.shape == (1, 1) and hasattr(labeled, 'dtype') and labeled.dtype.names:
                    # Access the structured array fields directly
                    labels_field = labeled['Labels'][0, 0]
                    data_field = labeled['Data'][0, 0]
                    
                    # Extract labels - they are wrapped in arrays
                    labels = self._extract_labels(labels_field)
                    
                    # Data shape is (n_markers, 4, n_frames)
                    # where the 4 values are [x, y, z, residual]
                    if data_field.ndim == 3:
                        n_markers, n_coords, n_frames = data_field.shape
                        
                        # Get frame rate from metadata if available
                        sampling_rate = trial.metadata.get('FrameRate', 100.0)
                        
                        # print(f"Extracting {n_markers} markers with {n_frames} frames")
                        
                        for i in range(n_markers):
                            name = labels[i] if i < len(labels) else f"Marker_{i+1}"
                            marker = MarkerData(
                                name=name,
                                x=data_field[i, 0, :],
                                y=data_field[i, 1, :],
                                z=data_field[i, 2, :],
                                sampling_rate=sampling_rate
                            )
                            trial.markers[name] = marker
                        
                        # print(f"Successfully extracted {len(trial.markers)} markers")
                                
        except (IndexError, KeyError, AttributeError) as e:
            print(f"Warning: Error extracting trajectories: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_qtm_analogs(self, analog_data: Any, trial: TrialData) -> None:
        """Extract analog signals from QTM Analog field."""
        if not isinstance(analog_data, np.ndarray) or analog_data.shape != (1, 1):
            return
        
        try:
            # Unwrap the (1,1) structure
            unwrapped = analog_data[0, 0]
            
            # Based on debug output:
            # [0] = BoardName
            # [1] = NrOfChannels
            # [2] = ChannelNumbers (1, 40)
            # [3] = Labels (1, 40)
            # [8] = Frequency
            # [9] = Data (40, 25560)
            
            if len(unwrapped) < 10:
                return
            
            # Extract channel numbers
            channel_numbers = unwrapped[2].flatten()
            
            # Extract labels
            labels_array = unwrapped[3]
            labels = self._extract_labels(labels_array)
            
            # Extract frequency
            frequency = float(unwrapped[8].flat[0]) if unwrapped[8].size > 0 else 1000.0
            
            # Extract data - shape is (n_channels, n_samples)
            data_array = unwrapped[9]
            
            if data_array.ndim == 2:
                n_channels, n_samples = data_array.shape
                
                for i in range(n_channels):
                    # Get channel name
                    if i < len(labels):
                        name = labels[i]
                    elif i < len(channel_numbers):
                        name = f"Channel_{int(channel_numbers[i])}"
                    else:
                        name = f"Analog_{i+1}"
                    
                    analog = AnalogData(
                        name=name,
                        data=data_array[i, :],
                        sampling_rate=frequency
                    )
                    trial.analogs[name] = analog
                    
        except (IndexError, KeyError, AttributeError) as e:
            print(f"Warning: Error extracting analog data: {e}")
    
    def _extract_qtm_force(self, force_data: Any, trial: TrialData) -> None:
        """Extract force plate data from QTM Force field."""
        if not isinstance(force_data, np.ndarray):
            return
        
        try:
            # Force data has shape (1, n_plates) where n_plates is number of force plates
            # Each plate has fields: ForcePlateName, Force, Moment, COP, Frequency, etc.
            
            if force_data.shape[0] == 1:
                n_plates = force_data.shape[1]
                
                for plate_idx in range(n_plates):
                    plate = force_data[0, plate_idx]
                    
                    if hasattr(plate, 'dtype') and plate.dtype.names:
                        # Extract plate name
                        if 'ForcePlateName' in plate.dtype.names:
                            name_array = plate['ForcePlateName']
                            if isinstance(name_array, np.ndarray) and name_array.size > 0:
                                plate_name = str(name_array.flat[0])
                            else:
                                plate_name = f"ForcePlate_{plate_idx+1}"
                        else:
                            plate_name = f"ForcePlate_{plate_idx+1}"
                        
                        # Extract force data (3, n_samples) - Fx, Fy, Fz
                        force_array = None
                        if 'Force' in plate.dtype.names:
                            force_field = plate['Force']
                            if isinstance(force_field, np.ndarray):
                                if force_field.shape == (1, 1):
                                    force_array = force_field[0, 0]
                                else:
                                    force_array = force_field
                        
                        # Extract moment data (3, n_samples) - Mx, My, Mz
                        moment_array = None
                        if 'Moment' in plate.dtype.names:
                            moment_field = plate['Moment']
                            if isinstance(moment_field, np.ndarray):
                                if moment_field.shape == (1, 1):
                                    moment_array = moment_field[0, 0]
                                else:
                                    moment_array = moment_field
                        
                        # Extract COP data (3, n_samples) - COPx, COPy, COPz
                        cop_array = None
                        if 'COP' in plate.dtype.names:
                            cop_field = plate['COP']
                            if isinstance(cop_field, np.ndarray):
                                if cop_field.shape == (1, 1):
                                    cop_array = cop_field[0, 0]
                                else:
                                    cop_array = cop_field
                        
                        # Extract frequency
                        frequency = 1000.0  # default
                        if 'Frequency' in plate.dtype.names:
                            freq_field = plate['Frequency']
                            if isinstance(freq_field, np.ndarray) and freq_field.size > 0:
                                frequency = float(freq_field.flat[0])
                        
                        # Create ForceData object if we have the required data
                        if force_array is not None and moment_array is not None and cop_array is not None:
                            force_obj = ForceData(
                                name=plate_name,
                                force=force_array,
                                moment=moment_array,
                                cop=cop_array,
                                sampling_rate=frequency
                            )
                            trial.forces[plate_name] = force_obj
                
                # print(f"Successfully extracted {len(trial.forces)} force plates")
                
        except (IndexError, KeyError, AttributeError) as e:
            print(f"Warning: Error extracting force data: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_markers(self, marker_data: Any, trial: TrialData) -> None:
        """Extract marker trajectories from QTM data structure."""
        if not isinstance(marker_data, np.ndarray):
            return
        
        # QTM exports markers in nested structure: marker_data[0][0][index]
        # Try to unwrap QTM structure first
        try:
            if marker_data.shape == (1, 1):
                # This is QTM format - unwrap it
                # Typical QTM structure:
                # [0][0][0] = something
                # [0][0][1] = something
                # [0][0][2] = marker labels
                # [0][0][3] = marker data (n_frames, 3*n_markers) or similar
                unwrapped = marker_data[0, 0]
                
                # Try to find marker labels and data
                marker_labels = None
                marker_coords = None
                sampling_rate = 100.0  # default
                
                # Inspect the unwrapped structure
                for idx in range(len(unwrapped)):
                    item = unwrapped[idx]
                    
                    # Look for marker labels (usually strings)
                    if isinstance(item, np.ndarray):
                        # Check if it's a string array (labels)
                        if item.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, or object strings
                            if marker_labels is None:
                                marker_labels = self._extract_labels(item)
                        
                        # Check if it's numeric data (coordinates)
                        elif item.dtype.kind in ['f', 'i']:  # float or int
                            if item.ndim >= 2 and item.shape[0] > 1:  # Has multiple samples
                                if marker_coords is None:
                                    marker_coords = item
                    
                    # Look for sampling rate
                    elif isinstance(item, (int, float, np.number)):
                        if 50 <= item <= 500:  # Reasonable marker sampling rate
                            sampling_rate = float(item)
                
                # Extract markers if we found coordinate data
                if marker_coords is not None:
                    self._parse_marker_coordinates(marker_coords, marker_labels, trial, sampling_rate)
                    return
                    
        except (IndexError, AttributeError):
            pass
        
        # Fallback to simple parsing if QTM structure doesn't match
        if marker_data.ndim == 3:
            # Assume format: (n_frames, 3, n_markers)
            n_frames, n_coords, n_markers = marker_data.shape
            for i in range(n_markers):
                marker = MarkerData(
                    name=f"Marker_{i+1}",
                    x=marker_data[:, 0, i],
                    y=marker_data[:, 1, i],
                    z=marker_data[:, 2, i]
                )
                trial.markers[marker.name] = marker
        elif marker_data.ndim == 2:
            # Assume format: (n_frames, 3) for single marker or (n_frames, 3*n_markers)
            if marker_data.shape[1] == 3:
                marker = MarkerData(
                    name="Marker_1",
                    x=marker_data[:, 0],
                    y=marker_data[:, 1],
                    z=marker_data[:, 2]
                )
                trial.markers[marker.name] = marker
            elif marker_data.shape[1] % 3 == 0:
                # Multiple markers in format (n_frames, 3*n_markers)
                n_markers = marker_data.shape[1] // 3
                for i in range(n_markers):
                    marker = MarkerData(
                        name=f"Marker_{i+1}",
                        x=marker_data[:, i*3],
                        y=marker_data[:, i*3 + 1],
                        z=marker_data[:, i*3 + 2]
                    )
                    trial.markers[marker.name] = marker
    
    def _extract_analogs(self, analog_data: Any, trial: TrialData) -> None:
        """Extract analog signals (EMG, force, etc.) from QTM data structure."""
        if not isinstance(analog_data, np.ndarray):
            return
        
        # QTM exports analogs in nested structure: analog_data[0][0][index]
        # According to debugging: [0][0][2] = channel numbers, [0][0][3] = labels, [0][0][4]+ = data
        try:
            if analog_data.shape == (1, 1):
                # This is QTM format - unwrap it
                unwrapped = analog_data[0, 0]
                
                # Extract components based on QTM structure
                channel_numbers = None
                channel_labels = None
                channel_data = []
                sampling_rate = 1000.0  # default
                
                # Parse the structure - typically:
                # [0] = some metadata
                # [1] = some metadata  
                # [2] = channel numbers
                # [3] = channel labels
                # [4+] = actual data arrays
                
                for idx in range(len(unwrapped)):
                    item = unwrapped[idx]
                    
                    if isinstance(item, np.ndarray):
                        # Check if it's channel numbers (small integers)
                        if item.dtype.kind in ['i', 'u'] and item.size > 0 and item.size < 100:
                            if np.all(item < 1000):  # Reasonable channel numbers
                                if channel_numbers is None:
                                    channel_numbers = item.flatten()
                        
                        # Check if it's labels (strings)
                        elif item.dtype.kind in ['U', 'S', 'O']:
                            if channel_labels is None:
                                channel_labels = self._extract_labels(item)
                        
                        # Check if it's actual signal data (large numeric arrays)
                        elif item.dtype.kind in ['f', 'i']:
                            if item.size > 100:  # Actual signal data should be large
                                channel_data.append(item.flatten())
                    
                    # Look for sampling rate
                    elif isinstance(item, (int, float, np.number)):
                        if 100 <= item <= 10000:  # Reasonable analog sampling rate
                            sampling_rate = float(item)
                
                # Create analog channels
                if channel_data:
                    for i, data in enumerate(channel_data):
                        # Get channel name
                        if channel_labels and i < len(channel_labels):
                            name = channel_labels[i]
                        elif channel_numbers is not None and i < len(channel_numbers):
                            name = f"Channel_{channel_numbers[i]}"
                        else:
                            name = f"Analog_{i+1}"
                        
                        analog = AnalogData(
                            name=name,
                            data=data,
                            sampling_rate=sampling_rate
                        )
                        trial.analogs[name] = analog
                    return
                    
        except (IndexError, AttributeError) as e:
            print(f"Warning: Error unwrapping QTM analog structure: {e}")
        
        # Fallback to simple parsing
        if analog_data.ndim == 2:
            n_samples, n_channels = analog_data.shape
            for i in range(n_channels):
                analog = AnalogData(
                    name=f"Analog_{i+1}",
                    data=analog_data[:, i]
                )
                trial.analogs[analog.name] = analog
        elif analog_data.ndim == 1:
            analog = AnalogData(
                name="Analog_1",
                data=analog_data
            )
            trial.analogs[analog.name] = analog
    
    def _extract_labels(self, label_array: np.ndarray) -> List[str]:
        """Extract string labels from numpy array."""
        labels = []
        flat = label_array.flatten()
        for item in flat:
            if isinstance(item, bytes):
                labels.append(item.decode('utf-8').strip())
            elif isinstance(item, str):
                labels.append(item.strip())
            elif isinstance(item, np.ndarray) and item.size == 1:
                # Sometimes labels are wrapped in another array
                inner = item.item()
                if isinstance(inner, bytes):
                    labels.append(inner.decode('utf-8').strip())
                elif isinstance(inner, str):
                    labels.append(inner.strip())
        return labels
    
    def _parse_marker_coordinates(self, coords: np.ndarray, labels: Optional[List[str]], 
                                   trial: TrialData, sampling_rate: float) -> None:
        """Parse marker coordinate data into MarkerData objects."""
        # Coords might be (n_frames, 3*n_markers) or (n_frames, n_markers, 3)
        if coords.ndim == 2 and coords.shape[1] % 3 == 0:
            # Format: (n_frames, 3*n_markers)
            n_markers = coords.shape[1] // 3
            for i in range(n_markers):
                name = labels[i] if labels and i < len(labels) else f"Marker_{i+1}"
                marker = MarkerData(
                    name=name,
                    x=coords[:, i*3],
                    y=coords[:, i*3 + 1],
                    z=coords[:, i*3 + 2],
                    sampling_rate=sampling_rate
                )
                trial.markers[name] = marker
        elif coords.ndim == 3:
            # Format: (n_frames, n_markers, 3) or (n_frames, 3, n_markers)
            if coords.shape[2] == 3:
                n_markers = coords.shape[1]
                for i in range(n_markers):
                    name = labels[i] if labels and i < len(labels) else f"Marker_{i+1}"
                    marker = MarkerData(
                        name=name,
                        x=coords[:, i, 0],
                        y=coords[:, i, 1],
                        z=coords[:, i, 2],
                        sampling_rate=sampling_rate
                    )
                    trial.markers[name] = marker
            else:
                n_markers = coords.shape[2]
                for i in range(n_markers):
                    name = labels[i] if labels and i < len(labels) else f"Marker_{i+1}"
                    marker = MarkerData(
                        name=name,
                        x=coords[:, 0, i],
                        y=coords[:, 1, i],
                        z=coords[:, 2, i],
                        sampling_rate=sampling_rate
                    )
                    trial.markers[name] = marker
    
    def debug_mat_structure(self, filepath: str) -> None:
        """Debug helper to inspect MAT file structure in detail."""
        mat_data = scipy.io.loadmat(filepath)
        trial_name = self._extract_trial_name(filepath, mat_data)
        
        print(f"=== Debugging MAT file structure ===")
        print(f"Trial name: {trial_name}\n")
        
        if trial_name in mat_data:
            main_data = mat_data[trial_name]
        else:
            metadata_keys = ['__header__', '__version__', '__globals__']
            data_keys = [k for k in mat_data.keys() if k not in metadata_keys]
            main_data = mat_data[data_keys[0]]
        
        print(f"Main data type: {type(main_data)}")
        print(f"Main data shape: {main_data.shape if hasattr(main_data, 'shape') else 'N/A'}")
        print(f"Main data dtype: {main_data.dtype if hasattr(main_data, 'dtype') else 'N/A'}")
        
        if hasattr(main_data, 'dtype') and main_data.dtype.names:
            print(f"\nField names: {main_data.dtype.names}\n")
            
            for field_name in main_data.dtype.names:
                print(f"--- Field: {field_name} ---")
                field_data = main_data[field_name][0, 0]
                print(f"  Type: {type(field_data)}")
                print(f"  Shape: {field_data.shape if hasattr(field_data, 'shape') else 'N/A'}")
                print(f"  Dtype: {field_data.dtype if hasattr(field_data, 'dtype') else 'N/A'}")
                
                # If it's a (1,1) array, unwrap it
                if hasattr(field_data, 'shape') and field_data.shape == (1, 1):
                    print(f"  Unwrapping (1,1) structure...")
                    unwrapped = field_data[0, 0]
                    print(f"  Unwrapped length: {len(unwrapped) if hasattr(unwrapped, '__len__') else 'N/A'}")
                    
                    if hasattr(unwrapped, '__len__'):
                        for idx in range(min(10, len(unwrapped))):  # Show first 10 items
                            item = unwrapped[idx]
                            print(f"    [{idx}] Type: {type(item)}, ", end="")
                            if hasattr(item, 'shape'):
                                print(f"Shape: {item.shape}, ", end="")
                            if hasattr(item, 'dtype'):
                                print(f"Dtype: {item.dtype}, ", end="")
                            if isinstance(item, np.ndarray) and item.size < 10:
                                print(f"Values: {item.flatten()}")
                            else:
                                print()
                print()
        
    def __repr__(self) -> str:
        return f"Subject(id='{self.subject_id}', trials={len(self.trials)})"
