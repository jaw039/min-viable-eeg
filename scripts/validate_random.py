import yaml

from src.loader import load_subject
from src.random_channels import (apply_random_budget,random_channels)


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

X, y, ch_names = load_subject(subject_id=1,config=config)

print("Original shape:")
print(X.shape)
print("\nTotal channels:")
print(len(ch_names))

for seed in [1, 2, 3]:
    selected = random_channels(ch_names, k=8, seed=seed)
    X_reduced, reduced_names = apply_random_budget(X, ch_names, k=8, seed=seed)

    print("\nSeed:", seed)
    print("Selected:", selected)
    print("Reduced shape:", X_reduced.shape)