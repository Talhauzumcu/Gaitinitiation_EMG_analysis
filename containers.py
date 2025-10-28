import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class MarkerData:
    """Data class for 3D marker trajectories."""
    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sampling_rate: float = 100.0
    
    def get_trajectory(self) -> np.ndarray:
        """Returns marker trajectory as (n_samples, 3) array."""
        return np.column_stack([self.x, self.y, self.z])
    
    def get_magnitude(self) -> np.ndarray:
        """Returns magnitude of 3D position."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    

@dataclass
class AnalogData:
    """Data class for analog signals (EMG, force plates, etc.)."""
    name: str
    data: np.ndarray
    sampling_rate: float = 1000.0
    unit: str = ""
    
    def get_data(self) -> np.ndarray:
        """Returns the analog signal data."""
        return self.data


@dataclass
class ForceData:
    """Data class for force plate data."""
    name: str
    force: np.ndarray  # (3, n_samples) - Fx, Fy, Fz
    moment: np.ndarray  # (3, n_samples) - Mx, My, Mz
    cop: np.ndarray  # (3, n_samples) - Center of Pressure x, y, z
    sampling_rate: float = 1000.0
    
    def __post_init__(self):
        self.transpose_data()
        
    def get_force_magnitude(self) -> np.ndarray:
        """Returns magnitude of force vector."""
        return np.sqrt(np.sum(self.force**2, axis=0))

    def transpose_data(self) -> None:
        """Transpose force, moment, and cop data to shape (n_samples, 3)."""
        self.force = self.force.T
        self.moment = self.moment.T
        self.cop = self.cop.T

@dataclass
class TrialData:
    """Data class for a single trial."""
    trial_name: str
    markers: Dict[str, MarkerData] = field(default_factory=dict)
    analogs: Dict[str, AnalogData] = field(default_factory=dict)
    forces: Dict[str, ForceData] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    events: Dict[str, int] = field(default_factory=dict)
    
    def get_marker_names(self) -> List[str]:
        """Returns list of marker names."""
        return list(self.markers.keys())
    
    def get_analog_names(self) -> List[str]:
        """Returns list of analog channel names."""
        return list(self.analogs.keys())
    
    def get_force_names(self) -> List[str]:
        """Returns list of force plate names."""
        return list(self.forces.keys())
    
    def get_marker(self, name: str) -> Optional[MarkerData]:
        """Get marker data by name."""
        return self.markers.get(name)
    
    def get_analog(self, name: str) -> Optional[AnalogData]:
        """Get analog data by name."""
        return self.analogs.get(name)
    
    def get_force(self, name: str) -> Optional[ForceData]:
        """Get force data by name."""
        return self.forces.get(name)
    
    def get_event(self, name: str) -> Optional[int]:
        """Get event time by name."""
        return self.events.get(name)