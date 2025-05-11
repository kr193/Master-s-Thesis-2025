#!/usr/bin/env python3
# File: src/preprocessing.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from src.dhs_modelling_functions_new import final_ds_droping_cols, fold_generator

def prepare_data(X_train, X_val, X_test, config):
    """
    Prepares the data by scaling numerical columns while leaving categorical columns unchanged.

    Parameters:
    - X_train: Training data (DataFrame).
    - X_val: Validation data (DataFrame).
    - X_test: Test data (DataFrame).
    - config: Dictionary containing configuration options.

    Returns:
    - X_train_scaled: Scaled training data (DataFrame).
    - X_val_scaled: Scaled validation data (DataFrame).
    - X_test_scaled: Scaled test data (DataFrame).
    """
    
    # identifying numerical and categorical columns
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Initialize copies to preserve structure
    X_train_scaled, X_val_scaled, X_test_scaled = X_train.copy(), X_val.copy(), X_test.copy()

    # Apply scaling to numerical columns if enabled in config
    if config['scale_numerical_data']:
        scaler = StandardScaler()

        # Scale numerical columns while keeping the index
        X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_val_scaled[numerical_columns] = scaler.transform(X_val[numerical_columns])
        X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Ensure numerical columns come first, followed by categorical columns
    ordered_columns = numerical_columns + categorical_columns
    X_train_scaled = X_train_scaled[ordered_columns]
    X_val_scaled = X_val_scaled[ordered_columns]
    X_test_scaled = X_test_scaled[ordered_columns]

    return X_train_scaled, X_val_scaled, X_test_scaled

def fold_generator_3_independent_indices(data, split_type, n_splits=2, verbose=1, val_size=0.2):
    """
    Generate indices for train, validation and test sets based on the specified split type.

    Parameters:
    data (DataFrame): The input dataset.
    split_type (str): The type of split - 'country', 'survey', or 'year'.
    n_splits (int): Number of splits/folds for the outer cross-validation.
    verbose (int): Level of verbosity.
    val_size (float): Proportion of the dataset to include in the validation split.
    """
    if split_type == 'survey':
        split_col = 'GEID'
    elif split_type == 'year':
        split_col = 'year of interview'
        # Ensure 'Meta; rounded year' column is created outside this function or create here based on logic provided
        data[split_col] = data.groupby('GEID')['year of interview'].transform(lambda x: round(x.mean()))
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

    # adjusts maximum n_splits based on the number of unique combinations
    if len(unique_combinations) < n_splits or n_splits == -1:
        n_splits = len(unique_combinations)
        if verbose:
            print(f'Adjusting n_splits to the length of unique combinations ({n_splits}) for', split_type)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_val_combinations, test_combinations in kf.split(unique_combinations):
        # splittig the train_val combinations into training and validation combinations
        train_combinations, val_combinations = train_test_split(train_val_combinations, test_size=val_size, random_state=42)
        
        # creating masks for training, validation, and test sets
        train_mask = data[split_col].isin(unique_combinations[train_combinations])
        val_mask = data[split_col].isin(unique_combinations[val_combinations])
        test_mask = data[split_col].isin(unique_combinations[test_combinations])
        
        # Get the indices for training, validation, and test sets
        train_indices = data[train_mask].index.values
        val_indices = data[val_mask].index.values
        test_indices = data[test_mask].index.values
        
        # Ensure train size is always larger than test
        assert len(train_indices) > len(test_indices), \
            f"Train size {len(train_indices)} should be larger than test size {len(test_indices)}"

        yield train_indices, val_indices, test_indices

def one_hot_encode_for_others(X_data, categorical_columns, encoder=None):
    """
    Apply OneHotEncoder to the specified categorical columns in the data for imputation methods.
    Ensures consistency between train, validation and test sets.

    Parameters:
    - X_data: Dataset containing categorical columns to encode.
    - categorical_columns: List of column names to encode.
    - encoder (OneHotEncoder): Pre-fitted encoder for consistency across datasets.
      If None, a new encoder will be fitted to the provided dataset.

    Returns:
    - X_data_encoded: DataFrame with categorical columns one-hot encoded.
    - encoder (OneHotEncoder): The fitted OneHotEncoder instance.
    """
    
    X_data = X_data.copy()  # avoid modifying original DataFrame
    
    # initializing and fitting encoder if not provided
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X_data[categorical_columns])

    # transforming categorical data into one-hot encoded format
    encoded_columns = encoder.transform(X_data[categorical_columns])
    encoded_column_names = encoder.get_feature_names_out(categorical_columns)
    # creating df with encoded categorical columns
    X_data_encoded = pd.DataFrame(encoded_columns, columns=encoded_column_names, index=X_data.index)

    # dropping original categorical columns
    X_data = X_data.drop(columns=categorical_columns)

    # concatenating numeric data with the encoded categorical data
    X_data_encoded = pd.concat([X_data, X_data_encoded], axis=1)

    return X_data_encoded, encoder

def consistent_one_hot_encode(train_data, train_noisy, val_data, test_data, categorical_columns):
    """
    One-hot encode the categorical columns consistently across train val and test sets.
    
    Parameters:
    - train_data, val_data, test_data: DataFrames containing train, val and test sets.
    - categorical_columns: List of column names to be one-hot encoded.
    
    Returns:
    - Tuple containing train_encoded, train_noisy_encoded, val_encoded, test_encoded: Encoded DataFrames for train, val and test sets.
    """
    
    # initializing OneHotEncoder and fitting only on the training dataset to avoid data leakage
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_data[categorical_columns])
    
    # transforming categorical columns across all datasets using the trained encoder
    train_encoded = encoder.transform(train_data[categorical_columns])
    train_noisy_encoded = encoder.transform(train_noisy[categorical_columns])
    val_encoded = encoder.transform(val_data[categorical_columns])
    test_encoded = encoder.transform(test_data[categorical_columns])
    
    encoded_column_names = encoder.get_feature_names_out(categorical_columns) # retrieving generated column names from encoder for clarity
    # converting the encoded arrays into dataframes with appropriate column names
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_column_names, index=train_data.index)
    train_noisy_encoded_df = pd.DataFrame(train_noisy_encoded, columns=encoded_column_names, index=train_noisy.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_column_names, index=val_data.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_column_names, index=test_data.index)
    
    # dropping the original categorical columns **without resetting index**
    train_data = train_data.drop(columns=categorical_columns)
    train_noisy = train_noisy.drop(columns=categorical_columns)
    val_data = val_data.drop(columns=categorical_columns)
    test_data = test_data.drop(columns=categorical_columns)

    # concatenating numeric and encoded categorical data
    train_encoded = pd.concat([train_data, train_encoded_df], axis=1)
    train_noisy_encoded = pd.concat([train_noisy, train_noisy_encoded_df], axis=1)
    val_encoded = pd.concat([val_data, val_encoded_df], axis=1)
    test_encoded = pd.concat([test_data, test_encoded_df], axis=1)
    
    return train_encoded, train_noisy_encoded, val_encoded, test_encoded
