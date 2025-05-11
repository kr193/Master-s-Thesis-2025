#!/usr/bin/env python3
# File: src/run_pipeline.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
# utility functions
from src.masking import apply_masking_for_cat_num
from src.config import RESULTS_DIR, GET_FINAL_IMPUTATIONS_DIR
from src.preprocessing import prepare_data, fold_generator_3_independent_indices
from src.utils import save_data, load_data, add_noise_to_categorical, separate_categorical_mask
from src.imputations import (run_autoencoder, run_dae, run_vae, run_gain, process_single_imputation, initial_knn_and_mode_imputed_data)

# -----------------------------------------------------------------
# Compute Data
# -----------------------------------------------------------------

def compute_data(final_imputations_dir_missing_ratio, missing_ratio, fold, repeat, X_train_final, X_val_final, X_test_final, config, imputers, masking):
    """
    Compute or load preprocessed datasets and imputed results for a specific fold, missing ratio and repetition using a defined 
    configuration and list of imputers. If not, it will be computed.
    """
    # sanitizing imputer names for consistent file naming
    standardized_imputers = {imputer: imputer.replace(' ', '_') for imputer in imputers}

    final_imputations_dir_missing_ratio = GET_FINAL_IMPUTATIONS_DIR(config)

    # paths for saving scaled, masked and imputed datasets
    scaled_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_scaled_final_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    mask_file = os.path.join(final_imputations_dir_missing_ratio, f'missing_mask_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    imputed_data_files = {imputer: os.path.join(final_imputations_dir_missing_ratio, f'{standardized_imputers[imputer]}_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl') for imputer in imputers}
    # additional intermediate files
    scaled_train_file = os.path.join(final_imputations_dir_missing_ratio, f'X_train_scaled_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    noisy_train_file = os.path.join(final_imputations_dir_missing_ratio, f'X_train_scaled_noisy_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    train_with_missing_file = os.path.join(final_imputations_dir_missing_ratio, f'X_train_with_missing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    test_with_missing_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_with_missing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    val_with_missing_file = os.path.join(final_imputations_dir_missing_ratio, f'X_val_with_missing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')

    # if the files do not exist, will be computed the necessary data
    print('Necessary files do not exist; they will be computed now from A to Z.')

    # optionally drop rows with too many missing values
    if config.get('process_nans', 'numerical_with_categorical') == 'numerical_with_categorical':

        # dropping the column named 'GEID' from X_train_final, X_val_final and X_test_final
        X_train_final = X_train_final.drop(columns=['GEID'], errors='ignore')
        X_val_final = X_val_final.drop(columns=['GEID'], errors='ignore')
        X_test_final = X_test_final.drop(columns=['GEID'], errors='ignore')
        # print(f"X_train_final", X_train_final.head())

    # performing scaling on train, validation and test data using the modified prepare_data function
    X_train_scaled_final, X_val_scaled_final, X_test_scaled_final = prepare_data(X_train_final, X_val_final, X_test_final, config)

    # separate numerical and categorical columns from X_train_scaled_final
    numerical_columns = X_train_scaled_final.select_dtypes(include=[np.number]).columns
    categorical_columns = X_train_scaled_final.select_dtypes(exclude=[np.number]).columns
    
    # adding noise only to the numerical columns of the scaled training data
    noise_factor = 0.2
    X_train_scaled_noisy_numerical = X_train_scaled_final[numerical_columns] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_scaled_final[numerical_columns].shape)
    X_train_scaled_noisy_numerical = np.clip(X_train_scaled_noisy_numerical, 0., 1.)

    # applying noise to the categorical columns
    X_train_scaled_noisy_categorical = add_noise_to_categorical(X_train_scaled_final.copy(), categorical_columns)

    # combining noisy numerical and original categorical data
    X_train_scaled_noisy_combined = pd.concat([X_train_scaled_noisy_numerical.reset_index(drop=True), X_train_scaled_noisy_categorical[categorical_columns].reset_index(drop=True)], axis=1)

    # ensuring both shapes match (they should have the same columns, only with noise added)
    assert X_train_scaled_noisy_combined.shape == X_train_scaled_final.shape, "Mismatch in shapes between noisy and original data!"

    # saving the scaled X_train and X_test
    with open(scaled_train_file, 'wb') as f:
        pickle.dump(X_train_scaled_final, f)
    with open(scaled_file, 'wb') as f:
        pickle.dump(X_test_scaled_final, f)
    
    print(f"Scaled X_test saved for fold {fold}, repetition {repeat}, and missing ratio {missing_ratio}")

    # applying masking to test and validation data
    X_train_with_missing, _ = apply_masking_for_cat_num(X_train_scaled_final.copy(), masking, missing_ratio)
    with open(train_with_missing_file, 'wb') as f:
        pickle.dump(X_train_with_missing, f)

    X_val_with_missing, _ = apply_masking_for_cat_num(X_val_scaled_final.copy(), masking, missing_ratio)
    with open(val_with_missing_file, 'wb') as f:
        pickle.dump(X_val_with_missing, f)
    print("X_val_scaled_final shape", X_val_scaled_final.shape)

    X_test_with_missing, missing_mask = apply_masking_for_cat_num(X_test_scaled_final.copy(), masking, missing_ratio)
    with open(test_with_missing_file, 'wb') as f:
        pickle.dump(X_test_with_missing, f)

    print("X_test_scaled_final", X_test_scaled_final.shape)

    with open(mask_file, 'wb') as f:
        pickle.dump(missing_mask, f)

    if isinstance(missing_mask, bool):
        raise ValueError("Expected `masking` to be the original missing mask DataFrame, but got a boolean. Please provide the correct mask.")

    # generating the categorical mask using the helper function
    categorical_mask = separate_categorical_mask(X_test_scaled_final, missing_mask)

    test_indices = X_test_with_missing.index  # Keep only test samples_di
    output_file_prefix = config['output_file_prefix']
    output_file = os.path.join(RESULTS_DIR, f'{output_file_prefix}_x_train_with_categorical_missingness_example.csv')

    X_train_with_missing.to_csv(output_file,index=False)
    
    print(f"Missing mask saved for fold {fold}, repetition {repeat}, and missing ratio {missing_ratio}")
    # print("X_test_with_missing", X_test_with_missing)
    
    # imputing validation and test data using initial KNN imputation
    X_train_scaled_one_hot_encoded, X_train_with_missing_one_hot_encoded, X_test_with_missing_one_hot_encoded, X_val_with_missing_one_hot_encoded, X_val_imputed_for_aes_one_hot_encoded, X_test_imputed_for_aes_one_hot_encoded, X_train_scaled_one_hot_encoded_for_aes, X_train_scaled_noisy_one_hot_encoded_for_aes = initial_knn_and_mode_imputed_data(X_train_scaled_final, X_train_scaled_noisy_combined, X_train_with_missing, X_val_with_missing, X_test_with_missing, fold, config, missing_ratio, repeat)
    output_file2= os.path.join(RESULTS_DIR, f'{output_file_prefix}_X_test_with_missing_one_hot_encoded_missingness_example.csv')

    X_test_with_missing_one_hot_encoded.to_csv(output_file2,index=False)

    imputed_data_store = {}

    original_train_columns = X_train_scaled_one_hot_encoded_for_aes.columns
    original_train_index = X_train_scaled_one_hot_encoded_for_aes.index
    
    original_val_columns = X_val_imputed_for_aes_one_hot_encoded.columns
    original_val_index = X_val_imputed_for_aes_one_hot_encoded.index
    
    original_test_columns = X_test_imputed_for_aes_one_hot_encoded.columns
    original_test_index = X_test_imputed_for_aes_one_hot_encoded.index

    X_train_scaled_one_hot_encoded_for_aes = np.nan_to_num(X_train_scaled_one_hot_encoded_for_aes)  
    X_val_imputed_for_aes_one_hot_encoded = np.nan_to_num(X_val_imputed_for_aes_one_hot_encoded)
    X_test_imputed_for_aes_one_hot_encoded = np.nan_to_num(X_test_imputed_for_aes_one_hot_encoded)

    X_train_scaled_one_hot_encoded_for_aes = pd.DataFrame(
        X_train_scaled_one_hot_encoded_for_aes, 
        columns=original_train_columns,  # storing the original column names before conversion
        index=original_train_index  # storing original index before conversion
    )
    
    X_val_imputed_for_aes_one_hot_encoded = pd.DataFrame(
        X_val_imputed_for_aes_one_hot_encoded, 
        columns=original_val_columns,
        index=original_val_index
    )
    
    X_test_imputed_for_aes_one_hot_encoded = pd.DataFrame(
        X_test_imputed_for_aes_one_hot_encoded, 
        columns=original_test_columns,
        index=original_test_index
    )

    # performing imputation for each imputer
    for imputer_name in imputers:
        if imputer_name == 'AE':
            tf.keras.backend.clear_session() 
            imputed_data = run_autoencoder(X_train_scaled_one_hot_encoded_for_aes, X_val_imputed_for_aes_one_hot_encoded, X_test_imputed_for_aes_one_hot_encoded, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, categorical_mask)
        elif imputer_name == 'DAE':
            tf.keras.backend.clear_session() 
            imputed_data = run_dae(X_train_scaled_noisy_one_hot_encoded_for_aes, X_val_imputed_for_aes_one_hot_encoded, X_test_imputed_for_aes_one_hot_encoded, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, categorical_mask)
        elif imputer_name == 'VAE':
            tf.keras.backend.clear_session() 
            imputed_data= run_vae(X_train_scaled_one_hot_encoded_for_aes, X_val_imputed_for_aes_one_hot_encoded, X_test_imputed_for_aes_one_hot_encoded, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, categorical_mask)
        elif imputer_name == 'GAIN':
            tf.keras.backend.clear_session()  
            imputed_data = run_gain(X_train_with_missing_one_hot_encoded, X_test_with_missing_one_hot_encoded, X_val_with_missing_one_hot_encoded, test_indices, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, categorical_mask)
        else:
            tf.keras.backend.clear_session()
            imputed_data = process_single_imputation(X_train_with_missing_one_hot_encoded, X_test_scaled_final, X_test_with_missing_one_hot_encoded, missing_ratio, fold, repeat, final_imputations_dir_missing_ratio, categorical_mask)
        
        # saving the imputed data for future use
        imputed_data_store[imputer_name] = pd.DataFrame(imputed_data, columns=X_test_imputed_for_aes_one_hot_encoded.columns, index=X_test_scaled_final.index)

    return X_test_scaled_final, missing_mask, imputed_data_store

# ---------------------------------------------------------
# Load or Compute Data Part
# ---------------------------------------------------------

# either loading precomputed data or computing imputed results for different folds and missing ratios
def load_or_compute_data_part(input_df, imputers, missing_ratios, config, masking):

    """
    Load or compute (and save) imputed data for each fold and missing ratio combination.
    
    Parameters:
    - input_df: The original input data.
    - imputers: List of imputation method names.
    - missing_ratios: List of missing data ratios to evaluate.
    - config: Configuration for processing and masking strategy.
    - masking: Type of masking to be applied to data ('MCAR', 'MAR').

    This function does not evaluate metrics, but only prepares and caches datasets.
    """
    final_imputations_dir_missing_ratio = GET_FINAL_IMPUTATIONS_DIR(config)
    
    fold_file_path = os.path.join(final_imputations_dir_missing_ratio, "all_generator_folds")
    fold_dir = os.path.join(fold_file_path, "folds")
    os.makedirs(fold_file_path, exist_ok=True)
    os.makedirs(fold_dir, exist_ok=True)

    # generating 5-fold cross-validation splits
    print("Using real data: Applying fold_generator_3_independent_indices")
    fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='survey', n_splits=5))

    # iterating over all folds and missing ratios to compute or load data
    for fold, (train_index, val_index, test_index) in enumerate(fold_gen):
        print(f"Processing fold {fold + 1} with masking={masking}")
        
        # file paths for train, validation, and test sets for this fold
        fold_dir_all = os.path.join(fold_dir, f'fold_{fold}')
        os.makedirs(fold_dir_all, exist_ok=True)
        
        train_file = os.path.join(fold_dir_all, 'X_train_final.pkl')
        val_file = os.path.join(fold_dir_all, 'X_val_final.pkl')
        test_file = os.path.join(fold_dir_all, 'X_test_final.pkl')

        # checking if train, validation and test sets already exist, if not compute and save them
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            print(f"Loading precomputed train, val, and test sets for fold {fold + 1}")
            X_train_final = load_data(train_file)
            X_val_final = load_data(val_file)
            X_test_final = load_data(test_file)
        else:
            print(f"Generating and saving train, val, and test sets for fold {fold + 1}")
            X_train_final, X_val_final, X_test_final = input_df.loc[train_index], input_df.loc[val_index], input_df.loc[test_index]
            print("At the very first glance of X_train_final", X_train_final)
            save_data(train_file, X_train_final)
            save_data(val_file, X_val_final)
            save_data(test_file, X_test_final)
            print(f"Train size: {len(X_train_final)}, Validation size: {len(X_val_final)}, Test size: {len(X_test_final)}")

        # for each missing ratio, process imputation
        for missing_ratio in missing_ratios:
            print(f"Processing missing ratio {missing_ratio}")

            # checking for must or required files
            scaled_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_scaled_final_{missing_ratio}_fold_{fold}_repeat_0.pkl')
            mask_file = os.path.join(final_imputations_dir_missing_ratio, f'missing_mask_{missing_ratio}_fold_{fold}_repeat_0.pkl')
            imputed_data_files = {imputer: os.path.join(final_imputations_dir_missing_ratio, f'{imputer}_imputed_{missing_ratio}_fold_{fold}_repeat_0.pkl') for imputer in imputers}

            # ensures all files for the current fold and missing ratio exist
            core_files_exist = (os.path.exists(scaled_file) and os.path.exists(mask_file) and all(os.path.exists(file) for file in imputed_data_files.values()))

            # only skipping if the files for the current fold and missing ratio exist
            if core_files_exist:
                print(f"All necessary files exist for fold {fold + 1} and missing ratio {missing_ratio}. Skipping imputation.")
            else:
                # if any file is missing, either load or compute data
                print(f"Computing data for fold {fold + 1} and missing ratio {missing_ratio}...")
                X_test_scaled_final, missing_mask, imputed_data_store = compute_data(
                    final_imputations_dir_missing_ratio, missing_ratio, fold, 0, X_train_final, X_val_final, X_test_final, config, imputers, masking)

    print("All data has been computed and saved as pickles.")