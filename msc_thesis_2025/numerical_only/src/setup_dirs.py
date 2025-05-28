#!/usr/bin/env python3
# File: src/setup_dirs.py

# importing required libraries
import os
from src.config_loader import load_config, find_project_root

# setting up at the beginning of the project (global config setup)
def setup_directories(config):
    project_root = find_project_root()

    # consistent base directory for synthetic data
    if config.get('use_synthetic_data'):
        # adding 'synthetic_data' folder explicitly like in get_final_imputations_dir
        base_dir = os.path.join(project_root, 'results', 'synthetic_data', f"synthetic_{config['correlation_type']}")
        output_images_base = os.path.join(project_root, 'output_images', 'synthetic_data', f"synthetic_{config['correlation_type']}")
    else:
        base_dir = os.path.join(project_root, 'results', config['process_nans'])
        output_images_base = os.path.join(project_root, 'output_images', config['process_nans'])

    # adding masking subdirectory
    masking_subdir = 'with_masking' if config.get('masking') else 'without_masking'

    # defining task directories
    task1_dir = os.path.join(base_dir, masking_subdir, 'final_methods_pkls', 'task1_final_imputations_missing_ratio')
    task2_dir = os.path.join(base_dir, masking_subdir, 'final_methods_pkls', 'task2_rmse_vs_num_features')
    task3_dir = os.path.join(base_dir, masking_subdir, 'final_methods_pkls', 'task3_time_vs_missing_ratio')
    output_images_dir = os.path.join(output_images_base, masking_subdir)

    # creating directories
    for path in [task1_dir, task2_dir, task3_dir, output_images_dir]:
        os.makedirs(path, exist_ok=True)
        print(f" Created or verified directory: {path}")

    # storing paths in config for easy access
    config['task1_dir'] = task1_dir
    config['task2_dir'] = task2_dir  
    config['task3_dir'] = task3_dir
    config['output_images_dir'] = output_images_dir

    return config



