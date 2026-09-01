"""
Central configuration with all experimental parameters.
This answers Q9 (Methods reconciliation) and locks the protocol.
"""
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DataConfig:
    """Dataset and preprocessing parameters."""
    # Dataset
    dataset_name: str = "PhysioNet EEG Motor Movement/Imagery"
    n_subjects: int = 109
    n_channels_total: int = 64
    
    # Preprocessing (from config.yaml - answers Q9)
    sampling_rate: int = 160  # Hz
    bandpass_low: float = 8.0  # Hz
    bandpass_high: float = 30.0  # Hz
    trial_window_start: float = 0.0  # seconds after cue
    trial_window_end: float = 4.0  # seconds after cue
    baseline_correction: float = -0.5  # seconds before cue
    
    # Split (subject-wise, seed 42 - from config.yaml)
    train_subjects: int = 74
    val_subjects: int = 16
    test_subjects: int = 16
    split_seed: int = 42
    
    # Classes (2-class - from config.yaml)
    classes: List[str] = None  # Will be set in __post_init__
    
    # Normalization
    normalize: bool = True
    norm_per_channel: bool = True  # Per-channel train-only normalization
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = ['left_fist', 'right_fist']  # 2-class from config.yaml

@dataclass
class ModelConfig:
    """EEGNet architecture parameters."""
    # Architecture
    n_classes: int = 2
    n_channels: int = 64
    n_samples: int = None  # Will be set from trial window
    n_temporal_filters: int = 8
    n_spatial_filters: int = 16
    n_separable_filters: int = 16
    kernel_length: int = 64
    dropout_rate: float = 0.25
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 64
    n_epochs: int = 100
    early_stopping_patience: int = 10
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    
    # Random seeds
    seeds: List[int] = None  # Will be set in __post_init__
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 101112]

@dataclass
class ExperimentConfig:
    """Experimental sweep parameters."""
    # Channel budgets
    budgets: List[int] = None  # Will be set in __post_init__
    
    # Selection modes (Q4)
    selection_modes: List[str] = None  # Will be set in __post_init__
    
    # Random controls (Q6)
    n_random_subsets: int = 20
    
    # Distillation (Q5)
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    distillation_budgets: List[int] = None  # Will be set in __post_init__
    
    # MVM thresholds (Q3)
    mvm_thresholds: List[float] = None  # Will be set in __post_init__
    
    def __post_init__(self):
        if self.budgets is None:
            self.budgets = [4, 6, 8, 12, 16, 32, 64]
        if self.selection_modes is None:
            self.selection_modes = ['ranked', 'random', 'sensorimotor']  # Q4
        if self.distillation_budgets is None:
            self.distillation_budgets = [4, 6, 8]
        if self.mvm_thresholds is None:
            self.mvm_thresholds = [0.85, 0.90, 0.95]  # Q3

# Global config instance
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
EXP_CONFIG = ExperimentConfig()

def get_config_summary() -> str:
    """Print config summary for Q9 Methods reconciliation."""
    summary = f"""
    ========================================
    EXPERIMENTAL PROTOCOL (Locked)
    ========================================
    
    Dataset: {DATA_CONFIG.dataset_name}
    Subjects: {DATA_CONFIG.n_subjects}
    Channels: {DATA_CONFIG.n_channels_total}
    
    Preprocessing:
      Sampling rate: {DATA_CONFIG.sampling_rate} Hz
      Bandpass: {DATA_CONFIG.bandpass_low}-{DATA_CONFIG.bandpass_high} Hz
      Window: {DATA_CONFIG.trial_window_start}-{DATA_CONFIG.trial_window_end}s post-cue
      Baseline: {DATA_CONFIG.baseline_correction}s pre-cue
      Classes: {DATA_CONFIG.classes}
    
    Split:
      Train: {DATA_CONFIG.train_subjects} subjects
      Val: {DATA_CONFIG.val_subjects} subjects
      Test: {DATA_CONFIG.test_subjects} subjects
      Seed: {DATA_CONFIG.split_seed}
      Type: Subject-wise
    
    Model:
      EEGNet with {MODEL_CONFIG.n_temporal_filters} temporal filters
      {MODEL_CONFIG.n_spatial_filters} spatial filters
      {MODEL_CONFIG.n_separable_filters} separable filters
      Dropout: {MODEL_CONFIG.dropout_rate}
      Learning rate: {MODEL_CONFIG.learning_rate}
      Batch size: {MODEL_CONFIG.batch_size}
      Epochs: {MODEL_CONFIG.n_epochs}
      Early stopping: {MODEL_CONFIG.early_stopping_patience}
    
    Experiment:
      Budgets: {EXP_CONFIG.budgets}
      Selection modes: {EXP_CONFIG.selection_modes}  # Q4
      Random subsets: {EXP_CONFIG.n_random_subsets}
      MVM thresholds: {EXP_CONFIG.mvm_thresholds}  # Q3
      Distillation budgets: {EXP_CONFIG.distillation_budgets}  # Q5
      Distillation T: {EXP_CONFIG.distillation_temperature}
      Distillation α: {EXP_CONFIG.distillation_alpha}
    
    Seeds: {MODEL_CONFIG.seeds}
    Selection: k* chosen on validation set (Q2)
    Test: evaluated once at selected k* (Q2)
    ========================================
    """
    return summary
