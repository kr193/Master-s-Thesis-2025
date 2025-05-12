#!/usr/bin/env python3
# src/config_loader.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import json

def load_config(config_file='config.json'):
    """
    Load configuration from a JSON file located in the project root (one level above src/).
    """
    # getting the directory where this script (config_loader.py) is located
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # going one level up to the project root
    project_root = os.path.dirname(current_file_dir)
    # constructing the full path to config.json
    config_path = os.path.join(project_root, config_file)
    # loading the JSON config
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def getting_project_root():
    """
    Detect the project root where config.json exists.
    """
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, 'config.json')):
        return current_dir
    else:
        # fallback: moving up from src/ to root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_file_dir)

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
    project_root = getting_project_root()

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



