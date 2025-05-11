#!/usr/bin/env python3
# File: src/setup_dirs.py
# optional script

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import os
from src.config import BASE_DIR

def setup_directories(config):
    """
    Sets up the base and masking-specific result directories.
    This structure is used to store output or pickle files from evaluations.
    """
     # using the main 'results' directory as base
    base_dir = os.path.join(BASE_DIR, 'results')
    print(f"Creating base directory: {base_dir}")
    os.makedirs(base_dir, exist_ok=True)
    # updating base_dir in config
    config['base_dir'] = base_dir

    # subdirectories for with and without masking
    with_masking_dir = os.path.join(base_dir, 'with_masking')
    without_masking_dir = os.path.join(base_dir, 'without_masking')

    os.makedirs(with_masking_dir, exist_ok=True)
    os.makedirs(without_masking_dir, exist_ok=True)

    return with_masking_dir, without_masking_dir




















