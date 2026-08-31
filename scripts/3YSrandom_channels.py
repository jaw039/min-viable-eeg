"""
Random channel subset generation with reproducibility.
Used for Q6: random controls.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class RandomChannelController:
    """Generate reproducible random channel subsets."""
    
    all_channels: List[str]
    base_seed: int = 12345
    
    def get_single_subset(self, k: int, seed: Optional[int] = None) -> Dict:
        """
        Generate a single random subset of k channels.
        
        Args:
            k: Number of channels to select
            seed: Random seed for reproducibility
        
        Returns:
            Dict with channels and metadata
        """
        if seed is None:
            seed = self.base_seed
        
        np.random.seed(seed)
        selected = np.random.choice(self.all_channels, k, replace=False)
        
        return {
            'channels': selected.tolist(),
            'seed': seed,
            'budget': k,
            'n_available': len(self.all_channels)
        }
    
    def get_subsets(self, k: int, n_subsets: int = 20) -> List[Dict]:
        """
        Generate n_subsets unique random subsets.
        
        Args:
            k: Number of channels per subset
            n_subsets: Number of subsets to generate
        
        Returns:
            List of subset dicts
        """
        subsets = []
        used_subsets = set()
        seed = self.base_seed
        
        while len(subsets) < n_subsets:
            np.random.seed(seed)
            selected = tuple(sorted(np.random.choice(
                self.all_channels, k, replace=False
            )))
            
            # Ensure uniqueness
            if selected not in used_subsets:
                used_subsets.add(selected)
                subsets.append({
                    'channels': list(selected),
                    'seed': seed,
                    'budget': k,
                    'subset_idx': len(subsets)
                })
            seed += 1
        
        return subsets
    
    def verify_subsets(self, subsets: List[Dict]) -> bool:
        """Verify all subsets are valid and unique."""
        if not subsets:
            return False
        
        k = subsets[0]['budget']
        
        # Check lengths
        for s in subsets:
            if len(s['channels']) != k:
                print(f"❌ Subset {s.get('subset_idx', '?')} has {len(s['channels'])} channels, expected {k}")
                return False
        
        # Check uniqueness
        channel_sets = [tuple(sorted(s['channels'])) for s in subsets]
        if len(channel_sets) != len(set(channel_sets)):
            print("❌ Duplicate subsets found")
            return False
        
        # Check all channels are valid
        all_valid = all(
            ch in self.all_channels 
            for s in subsets 
            for ch in s['channels']
        )
        if not all_valid:
            print("❌ Invalid channel names found")
            return False
        
        print(f"✓ All {len(subsets)} subsets verified (k={k})")
        return True