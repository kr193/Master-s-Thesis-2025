#!/usr/bin/env python3
# File: main.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

"""
Main script for coordinating the 'numerical with categorical' dataset imputation evaluation pipeline.
This script loads the configuration, initializes directories, loads data,
and runs the evaluation and visualization tasks.
"""

# importing required libraries
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from src.load_data import load_data
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from src.setup_dirs import setup_directories
from src.evaluation import handle_masking_and_evaluation
from src.config import BASE_DIR, MISSING_RATIOS, IMPUTERS

# ensures ./src is accessible for imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

def main():
    # short configuration for 'numerical with categorical' dataset 
    config = {
        'scale_numerical_data': True, # applying scaling to numerical columns
        'masking': False,              # set to True for MAR (masking applied) or False for MCAR (no masking);
                                      # can be toggled here to control which missingness scenario to evaluate
        'missingness_fraction': 0.3,  # for statistical computations, any ratio can be analysed, particularly
        'process_nans': 'numerical_with_categorical',  # numerical_with_categorical
        'drop_columns': ['GEID']      # columns to drop before processing 
    }

    # optional, initializing base, masking and subdirectories
    with_masking_dir, without_masking_dir = setup_directories(config) # only for creating the mar and mcar dirs beforehand
    # the output_file_prefix based on masking
    config['output_file_prefix'] = 'with_masking' if config['masking'] else 'without_masking'
    # list of imputation methods used in the additional work of the paper
    imputers = IMPUTERS
    # the missingness levels to evaluate (10% to 60%)
    missing_ratios = MISSING_RATIOS
    # main categorical with numerical data loading function, loads pickled or CSV file based on configuration
    input_df, initial_missingness = load_data(config)
    print(f"Loaded dataset with shape: {input_df.shape}")
    print(f"The Initial Missingness: {initial_missingness}")

    # --------------- Final evaluation for 'numerical with categorical' dataset ------------------
    handle_masking_and_evaluation(input_df, imputers, missing_ratios, config, masking=config['masking'])

    # #both options are run below by default for full comparison, but can be commented out
    # print("\n========= Starting Evaluation: With Masking =========")
    # # to enable masking-based evaluation (MAR)
    # handle_masking_and_evaluation(input_df, imputers, missing_ratios, config, masking=True)

    # print("\n========= Starting Evaluation: Without Masking =========")
    # # to enable masking-based evaluation (MCAR)
    # handle_masking_and_evaluation(input_df, imputers, missing_ratios, config, masking=False)
    print("\nAll evaluations completed for Numerical with Categorical dataset. Metrics saved.")

# running full pipeline 
if __name__ == "__main__":
    main()
