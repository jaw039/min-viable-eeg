"""
Channel budget management with frozen artifacts.
This ensures Q10: ranked path works and budgets are consistent.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class BudgetManager:
    """Manages channel budgets from frozen artifacts."""
    
    artifacts_dir: str = "data/frozen"
    
    def __post_init__(self):
        self.artifacts_path = Path(self.artifacts_dir)
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load frozen channel artifacts."""
        # Load channel ranking
        ranking_path = self.artifacts_path / "channel_ranking.json"
        if not ranking_path.exists():
            raise FileNotFoundError(
                f"Channel ranking not found at {ranking_path}. "
                "Run channel selection first (Jackie's task)."
            )
        
        with open(ranking_path) as f:
            self.ranking_data = json.load(f)
            self.ranked_channels = self.ranking_data.get('channels', [])
            self.importance_scores = self.ranking_data.get('scores', {})
        
        # Load budgets
        budgets_path = self.artifacts_path / "budgets.json"
        if budgets_path.exists():
            with open(budgets_path) as f:
                self.budgets = json.load(f)
        else:
            self.budgets = {}
    
    def top_k_channels(self, k: int) -> List[str]:
        """
        Return top k ranked channels.
        This is the RANKED selection path (Q10).
        """
        if k > len(self.ranked_channels):
            raise ValueError(f"k={k} > {len(self.ranked_channels)} available channels")
        return self.ranked_channels[:k].copy()
    
    def get_budget_channels(self, k: int, mode: str = 'ranked', 
                           random_seed: Optional[int] = None) -> List[str]:
        """
        Get channels for a given budget.
        
        Args:
            k: Number of channels
            mode: 'ranked', 'random', 'sensorimotor'
            random_seed: Seed for random mode
        
        Returns:
            List of channel names
        """
        if mode == 'ranked':
            return self.top_k_channels(k)
        
        elif mode == 'random':
            from src.random_channels import RandomChannelController
            controller = RandomChannelController(self.ranked_channels)
            if random_seed is None:
                random_seed = 42
            subset = controller.get_single_subset(k, seed=random_seed)
            return subset['channels']
        
        elif mode == 'sensorimotor':
            from src.sensorimotor import get_sensorimotor_channels
            return get_sensorimotor_channels(k)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'ranked', 'random', or 'sensorimotor'")
    
    def verify_consistency(self) -> bool:
        """
        Q10: Verify that budgets.json matches ranking.
        This prevents silent channel mismatches.
        """
        all_consistent = True
        
        if not self.budgets:
            print("⚠️ No budgets.json found - skipping consistency check")
            return True
        
        for k_str, stored_channels in self.budgets.items():
            k = int(k_str)
            if k > len(self.ranked_channels):
                continue  # Skip if budget larger than available channels
            
            ranked = self.top_k_channels(k)
            
            if sorted(stored_channels) != sorted(ranked):
                print(f"⚠️ Budget {k} mismatch!")
                print(f"  budgets.json: {stored_channels}")
                print(f"  ranking: {ranked}")
                all_consistent = False
        
        if all_consistent:
            print("✓ All budgets consistent with ranking")
        
        return all_consistent
    
    def get_full_channels(self) -> List[str]:
        """Return all available channels."""
        return self.ranked_channels.copy()
