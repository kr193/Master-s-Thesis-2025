#!/usr/bin/env python3
# src/config_loader.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import json
from pathlib import Path

def find_project_root():
    """
    moving up from this file's directory until it finds 'config.json'.
    Raises FileNotFoundError if it hits the filesystem root without finding it.
    """
    current = Path(__file__).resolve().parent

    # climbing until finding config.json or reaching the filesystem root
    while True:
        if (current / 'config.json').exists():
            return current
        if current.parent == current:
            # if reached the top and never saw config.json
            raise FileNotFoundError(f"Could not find '{root_marker}' in any parent of {__file__}")
        current = current.parent

def load_config(config_file: str = 'config.json') -> dict:
    """
    Load configuration from a JSON file located somewhere in the project root tree.
    Uses find_project_root() to locate the directory containing `config` file.
    """
    project_root = find_project_root(config_file)
    config_path = project_root / config_file
    with config_path.open('r') as f:
        return json.load(f)
        
# fetching or re-checking paths for task-specific use
def get_final_imputations_dir(config):
    """
    Construct and create result directories based on experiment settings.
    
    Returns:
        masking_dir (str): Directory for masking scenario.
        task1_dir (str): Directory for Task 1 results.
        task2_dir (str): Directory for Task 2 results.
        task3_dir (str): Directory for Task 3 results.
    """
    project_root = find_project_root()

    if config.get('use_synthetic_data'):
        base_dir = os.path.join(project_root, 'results', 'synthetic_data', f"synthetic_{config['correlation_type']}")
    else:
        base_dir = os.path.join(project_root, 'results', config['process_nans'])

    masking_dir = os.path.join(base_dir, 'with_masking' if config.get('masking') else 'without_masking')
    final_methods_dir = os.path.join(masking_dir, 'final_methods_pkls')

    task1_dir = os.path.join(final_methods_dir, 'task1_final_imputations_missing_ratio')
    task2_dir = os.path.join(final_methods_dir, 'task2_rmse_vs_num_features')
    task3_dir = os.path.join(final_methods_dir, 'task3_time_vs_missing_ratio')

    for path in [final_methods_dir, task1_dir, task2_dir, task3_dir]:
        os.makedirs(path, exist_ok=True)

    return masking_dir, task1_dir, task2_dir, task3_dir



