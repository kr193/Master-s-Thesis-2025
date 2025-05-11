#!/usr/bin/env python3
# File: src/load_data.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pandas as pd
from src.config import BASE_DIR
from src.utils import calculate_initial_missingness

# load_data function
def load_data(config):
    """
    Loads the 'numerical with categorical' DHS's tabular dataset from CSV to examine a real-world scenario.
    """
    df_path = os.path.join(BASE_DIR, 'data', 'numerical_with_categorical_dataset_complete_dataset.csv')
    df = pd.read_csv(df_path)

    if config['process_nans'] == 'numerical_with_categorical':
        return df, calculate_initial_missingness(df)
    
    return df, calculate_initial_missingness(df)
