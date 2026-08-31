#!/usr/bin/env python
"""
Q7: Reproducibility Anchors
Run before the main sweep to verify the pipeline works.
Three checks that must pass before trusting results.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict

# Import from src
from src.config import DATA_CONFIG, MODEL_CONFIG, EXP_CONFIG
from src.models import EEGNet
from src.budget import BudgetManager
from src.evaluation import calculate_kappa
from src.experiment_runner import ExperimentRunner

def check_1_published_comparison() -> Dict:
    """
    Check 1: Does our 8-channel performance match published ranges?
    Expected: κ ≈ 0.60-0.70 for 8-channel PhysioNet with EEGNet
    """
    print("\n" + "="*60)
    print("Q7 Check 1: Published Comparison")
    print("-"*50)
    
    # Load 8-channel configuration
    budget_mgr = BudgetManager()
    channels = budget_mgr.top_k_channels(8)
    
    # Quick