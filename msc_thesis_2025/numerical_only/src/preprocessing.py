#!/usr/bin/env python3
# File: src/preprocessing.py

import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.config_loader import load_config
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from src.dhs_modelling_functions_new import final_ds_droping_cols, fold_generator

def apply_masking(X_data, masking, missingness_fraction):
    '''
    Applies either MAR-style masking (grouped by base column) or MCAR-style random masking.
    Returns the masked data and corresponding mask matrix.
    '''
    X_data=X_data.copy()

    missing_mask = np.zeros(X_data.shape, dtype=bool)
    if masking:
        found_base_col = set()
        for col in X_data.columns:
            base_col = col.rsplit(':', 1)[0]
            if base_col in found_base_col:
                continue
            found_base_col.add(base_col)
            non_missing_indices = X_data[col].dropna().index
            random_sample = np.random.choice(non_missing_indices, size=int(np.ceil(missingness_fraction * len(non_missing_indices))), replace=False)
            local_indices = X_data.index.get_indexer(random_sample)
            columns_with_same_base = [c for c in X_data.columns if c.startswith(base_col)]
            for col2 in columns_with_same_base:
                missing_mask[local_indices, X_data.columns.get_loc(col2)] = True
                X_data.loc[random_sample, col2] = np.nan
    else:
        for col in X_data.columns:
            non_missing_indices = X_data[col].dropna().index
            random_sample = np.random.choice(non_missing_indices, size=int(np.ceil(missingness_fraction * len(non_missing_indices))), replace=False)
            local_indices = X_data.index.get_indexer(random_sample)
            missing_mask[local_indices, X_data.columns.get_loc(col)] = True
            X_data.loc[random_sample, col] = np.nan
    return X_data, missing_mask

def prepare_data(X_train, X_val, X_test, config):
    """
    Prepares training, validation, and test datasets by applying preprocessing strategies:
    - Drops user-defined columns if the user selects 'drop_all_nans' from the prompt.
    - Removes specific unimportant columns to streamline data processing.
    - Applies StandardScaler normalization if 'scale_numerical_data' is set to True.
    """
    # only drop columns if 'process_nans' == 'drop_all_nans'
    # the raw fetched DHS dataset has categorical columns also like "Meta; adm0_gaul", "Meta; GEID_init"
    # if user selects this 'drop_all_nans' option, to ensure the dataset to be full numerical dataset, the mentioned    cat. columns are dropped using the 'drop_columns' config.
    process_nans = config.get('process_nans', '') 
    
    # for 'drop_all_nans' specifically, drop predefined columns if listed
    if process_nans == 'drop_all_nans' and 'drop_columns' in config:
        X_train = X_train.drop(columns=config['drop_columns'], errors='ignore')
        X_val = X_val.drop(columns=config['drop_columns'], errors='ignore')
        X_test = X_test.drop(columns=config['drop_columns'], errors='ignore')

    # apply scaling
    if config.get('scale_numerical_data', True):
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test

    return X_train_scaled, X_val_scaled, X_test_scaled

def prepare_data_for_feature_evaluation(X_train, X_val, X_test, config):
    """
    - Applies StandardScaler normalization if 'scale_numerical_data' is set to True.
    """
    if config['scale_numerical_data']:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def fold_generator_3_independent_indices(data, split_type, n_splits=5, verbose=1, val_size=0.2):
    """
    Generate indices for train, validation and test sets based on the specified split type.

    Parameters:
    data (DataFrame): The input dataset.
    split_type (str): The type of split - 'country', 'survey', or 'year'.
    n_splits (int): Number of splits/folds for the outer cross-validation.
    verbose (int): Level of verbosity.
    val_size (float): Proportion of the dataset to include in the validation split.
    """
    if split_type == 'country':
        split_col = 'Meta; adm0_gaul'
    elif split_type == 'survey':
        split_col = 'Meta; GEID_init'
    elif split_type == 'year':
        split_col = 'Meta; rounded year'
        # Ensure 'Meta; rounded year' column is created outside this function or create here based on logic provided
        data[split_col] = data.groupby('Meta; GEID_init')['Meta; year'].transform(lambda x: round(x.mean()))
    elif split_type == 'unconditional':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_val_idx, test_idx in kf.split(data):
            # Split the train_val indices into training and validation indices
            train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=42)
            yield data.index[train_idx], data.index[val_idx], data.index[test_idx]
        return
    else:
        raise ValueError(f'Invalid split_type: {split_type}')

    unique_combinations = data[split_col].drop_duplicates().values

    # Adjust maximum n_splits based on the number of unique combinations
    if len(unique_combinations) < n_splits or n_splits == -1:
        n_splits = len(unique_combinations)
        if verbose:
            print(f'Adjusting n_splits to the length of unique combinations ({n_splits}) for', split_type)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_val_combinations, test_combinations in kf.split(unique_combinations):
        # Split the train_val combinations into training and validation combinations
        train_combinations, val_combinations = train_test_split(train_val_combinations, test_size=val_size, random_state=42)
        
        # Create masks for training, validation, and test sets
        train_mask = data[split_col].isin(unique_combinations[train_combinations])
        val_mask = data[split_col].isin(unique_combinations[val_combinations])
        test_mask = data[split_col].isin(unique_combinations[test_combinations])
        
        # Get the indices for training, validation, and test sets
        train_indices = data[train_mask].index.values
        val_indices = data[val_mask].index.values
        test_indices = data[test_mask].index.values
        
        # Yielding the indices for train, validation and test sets
        yield train_indices, val_indices, test_indices