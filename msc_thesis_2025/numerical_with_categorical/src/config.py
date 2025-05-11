#!/usr/bin/env python3
# File: src/config.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# common parameters
MISSING_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
IMPUTERS = ['Mean', 'KNN', 'MICE_Ridge', 'AE', 'DAE', 'VAE', 'GAIN']

def GET_FINAL_IMPUTATIONS_DIR(config):
    """
    Returns the path to the final_imputations_dir_missing_ratio based on masking value in config.
    Automatically creates the path if not present.
    """
    # directories for saving the data
    # correct subdirectory based on masking flag
    masking_dir = os.path.join(RESULTS_DIR, 'with_masking' if config.get('masking') else 'without_masking')
    # subdirectories for saving experiment outputs
    final_methods_dir = os.path.join(masking_dir, 'final_methods_pkls')
    final_imputations_dir = os.path.join(final_methods_dir, 'task1_final_imputations_missing_ratio')
    # ensure the final directory exists
    os.makedirs(final_imputations_dir, exist_ok=True)
    return final_imputations_dir



