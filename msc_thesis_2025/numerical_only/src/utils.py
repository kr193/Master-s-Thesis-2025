#!/usr/bin/env python3
# File: src/utils.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pickle
import string
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import tensorflow.keras.backend as K
from src.config_loader import get_final_imputations_dir
from sklearn.metrics import mean_squared_error, r2_score

# for root mean squared error (RMSE) calculation
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# to calculate initial missingness
def calculate_initial_missingness(df):
    total_elements = df.size
    missing_elements = df.isnull().sum().sum()
    return missing_elements / total_elements

def drop_all_nans(df):
    non_missing_indices = {}
    data_with_missing = df.copy()
    for col in data_with_missing.columns:
        non_missing_indices[col] = data_with_missing[col].dropna().index
    
    values_df = pd.DataFrame()
    for col, indices in non_missing_indices.items():
        values_df[col] = data_with_missing.loc[indices, col].reset_index(drop=True)
    
    actual_values_df = values_df.dropna()
    rows = actual_values_df.shape
    print(f"Shape after dropping all NaNs: {rows}")
    
    return actual_values_df

# to load imputed data from pickle files
def load_imputed_data(imputer_name, missing_ratio, fold, config):
    _, final_imputations_dir_missing_ratio, _, _ = get_final_imputations_dir(config)

    # defining the expected file path
    filename = os.path.join(
        final_imputations_dir_missing_ratio, f"{imputer_name}_imputed_{missing_ratio}_fold_{fold}_repeat_0.pkl")

    # if the file exists and print the path for debugging
    print(f"Trying to load imputed data from: {filename}")
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    print(f"File does not exist: {filename}")
    return None

# to load imputed data from pickle files for Task2: feature evaluation
def load_imputed_data_feature_evaluation(imputer_name, n_features, fold, config):
    """
    Load imputed data from pickle files based on the number of features and fold.

    Parameters:
    - imputer_name: Name of the imputer method.
    - n_features: Number of features used in the imputation.
    - fold: Current fold number.

    Returns:
    - imputed_data: The imputed data or None if the file doesn't exist.
    """
    masking_dir, final_imputations_dir_missing_ratio, feature_eval_dir, task3_time_dir = get_final_imputations_dir(config)

    # defining the filename based on the number of features and fold
    filename = os.path.join(feature_eval_dir, f"{imputer_name}_imputed_{n_features}_features_fold_{fold}_repeat_0.pkl")
    
    # try loading the imputed data from the specified file
    print(f"Trying to load imputed data from: {filename}")
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    print(f"File does not exist: {filename}")
    return None

# to load time data from pickle files
def load_time_data(imputer_name, missing_ratio, fold, config, time_type='train'):
    _, _, _, task3_time_dir = get_final_imputations_dir(config)
    
    if time_type == 'train':
        filename = os.path.join(task3_time_dir, f"{imputer_name}_train_timing_{missing_ratio}_fold_{fold}_repeat_0.pkl")
    else:
        filename = os.path.join(task3_time_dir, f"{imputer_name}_test_timing_{missing_ratio}_fold_{fold}_repeat_0.pkl")
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return mse, rmse, r2, corr

# to save time related metrics into excel
def save_time_metrics_to_excel(time_metrics, config):
    """
    Save time metrics to an Excel file with an appropriate prefix.
    """
    df_time_metrics = pd.DataFrame(time_metrics)

    output_file_prefix = config['output_file_prefix']
    output_images_dir = config['output_images_dir']

    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']

    output_file = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_time_vs_missing_ratio.xlsx")
    df_time_metrics.to_excel(output_file, index=False)
    print(f"Time metrics saved to {output_file}")

# to extract values 
def extract_values_using_mask(data, mask):
    if isinstance(data, pd.DataFrame):
        if data.shape != mask.shape:
            raise ValueError(f"Data shape: {data.shape}, Mask shape: {mask.shape}")
        # extracting values where the mask is True
        extracted_values = data[mask]
        return extracted_values.dropna()  # ensuring only non-NaN rows are kept

    elif isinstance(data, np.ndarray):
        if data.shape != mask.shape:
            raise ValueError("The shape of data and mask do not match.")
        return data[mask]

    else:
        raise TypeError("Data must be either a pandas DataFrame or numpy array.")

# def extract_values_using_mask(data, mask):
#     """
#     Extracts values from 'data' at positions where 'mask' is True.
#     Both data and mask must have the same shape.
#     Returns a 1D numpy array of values.
#     """
#     if isinstance(data, pd.DataFrame):
#         data = data.values
#     elif not isinstance(data, np.ndarray):
#         raise TypeError("Data must be either a pandas DataFrame or numpy array.")

#     if data.shape != mask.shape:
#         raise ValueError(f"Data shape: {data.shape}, Mask shape: {mask.shape}")

#     return data[mask]
        
# to load pickles
def safe_load_pickle(file_path):
    """
    Safely load a pickle file and handle any exceptions.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as e:
        print(f"Error loading pickle file: {file_path}, error: {e}")
        return None

# to save pickles
def save_pickle(file_path, data):
    """
    Utility function to save data as a pickle file.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

def check_missing_imputers(imputed_data_files):
    """
    Check if any imputer result files are missing.

    Parameters:
    - imputed_data_files: Dictionary of imputer name and corresponding file paths.

    Returns:
    - missing_imputers: List of imputer names whose result files are missing.
    """
    missing_imputers = [imputer for imputer, file in imputed_data_files.items() if not os.path.exists(file)]
    return missing_imputers

