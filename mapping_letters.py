import os, re

# 1. List only your class folders
folders = [d for d in os.listdir("Test") if d.startswith("character_")]  # turn1search3

# 2. Sort by the numeric component
folders_sorted = sorted(
    folders,
    key=lambda name: int(re.match(r"character_(\d+)_", name).group(1))  # turn1search2
)

print(folders_sorted)  # Copy this output for the next step
