"""
Sensorimotor channel restriction for Q4 control experiment.
This answers: Is the signal motor or cue-correlated?
"""
from typing import List

# Core sensorimotor strip channels (based on 10-10 system)
# These cover the motor cortex region
SENSORIMOTOR_CHANNELS = [
    'C3', 'C4', 'Cz',           # Central
    'FCz', 'CPz',               # Frontocentral/centroparietal midline
    'C1', 'C2',                 # Central medial
    'CP1', 'CP2',               # Centroparietal
    'FC1', 'FC2',               # Frontocentral
    'C5', 'C6',                 # Central lateral
    'CP3', 'CP4',               # Centroparietal lateral
    'FC3', 'FC4'                # Frontocentral lateral
]

# Expanded sensorimotor set
SENSORIMOTOR_EXPANDED = SENSORIMOTOR_CHANNELS + [
    'Fz', 'Pz',                 # Midline
    'AFz', 'POz',               # Extended midline
    'F3', 'F4', 'P3', 'P4'      # Lateral
]

def get_sensorimotor_channels(k: int, expanded: bool = False) -> List[str]:
    """
    Return k sensorimotor channels, ranked by relevance.
    
    Args:
        k: Number of channels to return
        expanded: If True, use expanded sensorimotor set
    
    Returns:
        List of k channel names
    """
    pool = SENSORIMOTOR_EXPANDED if expanded else SENSORIMOTOR_CHANNELS
    return pool[:k].copy()

def is_sensorimotor(channel: str) -> bool:
    """Check if a channel is in the sensorimotor pool."""
    return channel in SENSORIMOTOR_CHANNELS

def restrict_to_sensorimotor(ranked_channels: List[str], k: int) -> List[str]:
    """
    Take a full ranking and return top k channels that are in sensorimotor pool.
    
    Used when comparing unrestricted vs. sensorimotor-restricted.
    """
    sensorimotor_ranked = [ch for ch in ranked_channels if is_sensorimotor(ch)]
    return sensorimotor_ranked[:k]

def get_sensorimotor_description() -> str:
    """Return description for methods section."""
    return (
        f"Sensorimotor-restricted channels include {len(SENSORIMOTOR_CHANNELS)} "
        "electrodes covering the central and central-parietal regions (C3, C4, Cz, "
        "FCz, CPz, and surrounding electrodes). This set was based on neurophysiological "
        "knowledge of motor cortex involvement in motor imagery."
    )
