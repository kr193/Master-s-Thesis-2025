#!/usr/bin/env python3
# File: src/run_pipeline.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from src.utils import *
from src.preprocessing import *
from src.single_imputation import *
from src.initial_imputation import *
from src.deep_learning_methods import *
from src.config_loader import get_final_imputations_dir
from src.helper_functions import select_process_nans_option

# -----------------------------------------------------------------
# Compute Data for Task1 and Task3
# -----------------------------------------------------------------

def compute_data(final_imputations_dir_missing_ratio, missing_ratio, fold, repeat, X_train_final, X_val_final, X_test_final, config, imputers, masking):
    """
    Check if the necessary data files (X_test_scaled_final, missing mask, imputed data) already exist. If not, compute them.
    """
    masking_dir, final_imputations_dir_missing_ratio, feature_eval_dir, task3_time_dir = get_final_imputations_dir(config)

    standardized_imputers = {imputer: imputer.replace(' ', '_') for imputer in imputers}
    
    scaled_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_scaled_final_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    mask_file = os.path.join(final_imputations_dir_missing_ratio, f'missing_mask_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    imputed_data_files = {imputer: os.path.join(final_imputations_dir_missing_ratio, f'{standardized_imputers[imputer]}_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl') for imputer in imputers}
    scaled_train_file = os.path.join(final_imputations_dir_missing_ratio, f'X_train_scaled_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    noisy_train_file = os.path.join(final_imputations_dir_missing_ratio, f'X_train_scaled_noisy_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    train_with_missing_file = os.path.join(final_imputations_dir_missing_ratio, f'X_train_with_missing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    test_with_missing_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_with_missing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    val_with_missing_file = os.path.join(final_imputations_dir_missing_ratio, f'X_val_with_missing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    val_imputed_file = os.path.join(final_imputations_dir_missing_ratio, f"X_val_knn_imputed_aes_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    test_imputed_file = os.path.join(final_imputations_dir_missing_ratio, f"X_test_knn_imputed_aes_{missing_ratio}_fold_{fold}_{missing_ratio}_repeat_{repeat}.pkl")
    
    train_time_files = {imputer: os.path.join(task3_time_dir, f"{imputer}_train_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl") for imputer in imputers}
    test_time_files = {imputer: os.path.join(task3_time_dir, f"{imputer}_test_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl") for imputer in imputers}
    
    # if the files do not exist, compute the necessary data
    print('Necessary files do not exist; they will be computed now from A to Z.')

    selected_option = config['process_nans']
    assert isinstance(selected_option, str), "Expected 'process_nans' to be a string"

    if not config.get('use_synthetic_data', False):
        if selected_option in ['numerical_only_drop_20_percentage_nans', 'keep_all_numerical']:

            X_train_final = X_train_final.select_dtypes(include=[np.number])
            X_val_final = X_val_final.select_dtypes(include=[np.number])
            X_test_final = X_test_final.select_dtypes(include=[np.number])
    
            # performing KNN imputation or load saved imputed data
            X_train_final, X_val_final, X_test_final = first_knn_imp_for_numerical_data_with_missing(X_train_final, X_val_final, X_test_final, fold, masking_dir, missing_ratio, repeat, config)

    # performing scaling on train, validation and test data
    X_train_scaled_final, X_val_scaled_final, X_test_scaled_final = prepare_data(X_train_final, X_val_final, X_test_final, config)
    
    print("X_train_scaled_final Columns:", X_train_scaled_final.shape)
    print("X_train_scaled_final Columns:", X_train_scaled_final.head(10))

    noise_factor = 0.2
    X_train_scaled_noisy = X_train_scaled_final + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_scaled_final.shape)
    X_train_scaled_noisy = np.clip(X_train_scaled_noisy, 0., 1.)  
    
    # saving the scaled X_train and X_test
    with open(scaled_train_file, 'wb') as f:
        pickle.dump(X_train_scaled_final, f)
    with open(scaled_file, 'wb') as f:
        pickle.dump(X_test_scaled_final, f)
    print(f"Scaled X_test saved for fold {fold}, repetition {repeat}, and missing ratio {missing_ratio}")

    # applying masking to test and validation data
    X_train_with_missing, _ = apply_masking(X_train_scaled_final.copy(), masking, missing_ratio)
    with open(train_with_missing_file, 'wb') as f:
        pickle.dump(X_train_with_missing, f)

    X_val_with_missing, _ = apply_masking(X_val_scaled_final.copy(), masking, missing_ratio)
    with open(val_with_missing_file, 'wb') as f:
        pickle.dump(X_val_with_missing, f)

    X_test_with_missing, missing_mask = apply_masking(X_test_scaled_final.copy(), masking, missing_ratio)
    with open(test_with_missing_file, 'wb') as f:
        pickle.dump(X_test_with_missing, f)

    # output_file="x_train_mar_missingness_example.csv"
    # X_train_with_missing.to_csv(output_file,index=False)

    with open(mask_file, 'wb') as f:
        pickle.dump(missing_mask, f)
    print(f"Missing mask saved for fold {fold}, repetition {repeat}, and missing ratio {missing_ratio}")

    X_val_imputed_final, X_test_imputed_final = initial_knn_imputed_data_for_aes(X_train_scaled_final, X_val_with_missing, X_test_with_missing, fold, masking_dir, missing_ratio, repeat, config)
    with open(val_imputed_file, 'wb') as f:
        pickle.dump(X_val_imputed_final, f)
    with open(test_imputed_file, 'wb') as f:
        pickle.dump(X_test_imputed_final, f)
        
    imputed_data_store = {}
    train_time_data_store = {}
    test_time_data_store = {}
    # performing imputation for each imputer
    for imputer_name in imputers:
        # print(f"Processing first time the {imputer_name} for fold {fold} and repetition {repeat}")
        if imputer_name == 'AE':
            tf.keras.backend.clear_session() 
            imputed_data, training_time_imputer, test_time_imputer = run_autoencoder(X_train_scaled_final, X_test_imputed_final, X_val_imputed_final, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, config)
        elif imputer_name == 'DAE':
            tf.keras.backend.clear_session() 
            imputed_data, training_time_imputer, test_time_imputer = run_dae(X_train_scaled_noisy, X_test_imputed_final, X_val_imputed_final, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, config)
        elif imputer_name == 'VAE':
            tf.keras.backend.clear_session() 
            imputed_data, training_time_imputer, test_time_imputer = run_vae(X_train_scaled_final, X_test_imputed_final, X_val_imputed_final, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, config)
        elif imputer_name == 'GAIN':
            tf.keras.backend.clear_session()  
            imputed_data, training_time_imputer, test_time_imputer = run_gain(X_train_scaled_final, X_test_with_missing, X_val_with_missing, final_imputations_dir_missing_ratio, fold, repeat, missing_ratio, config)
        else:
            tf.keras.backend.clear_session()
            imputed_data, training_time_imputer, test_time_imputer = process_single_imputation(X_train_with_missing, X_test_scaled_final, X_test_with_missing, missing_ratio, fold, repeat, final_imputations_dir_missing_ratio, config)

        # saving the imputed data for future use
        imputed_data_store[imputer_name] = pd.DataFrame(imputed_data, columns=X_test_scaled_final.columns)
        
        train_time_data_store[imputer_name] = training_time_imputer
        test_time_data_store[imputer_name] = test_time_imputer

    return X_test_scaled_final, missing_mask, imputed_data_store, train_time_data_store, test_time_data_store

# ---------------------------------------------------------
# Load or Compute Data Part for Task1 and Task3
# ---------------------------------------------------------

def load_or_compute_data_part(input_df, imputers, missing_ratios, config, masking):
    """
    Load or compute data (save pickles), without performing metrics evaluation.
    """
    # ensuring 'process_nans' is set (prompt if missing)
    if not config.get('use_synthetic_data', False):
        if 'process_nans' not in config or not isinstance(config['process_nans'], str):
            config = select_process_nans_option(config)
            print(f" The selected process_nans: {config['process_nans']}")
    
    masking_dir, final_imputations_dir_missing_ratio, feature_eval_dir, task3_time_dir = get_final_imputations_dir(config)
    
    fold_file_path = os.path.join(final_imputations_dir_missing_ratio, "all_generator_folds")
    fold_dir = os.path.join(fold_file_path, "folds")
    os.makedirs(fold_file_path, exist_ok=True)
    os.makedirs(fold_dir, exist_ok=True)

    # generates 5-fold cross-validation splits
    # determines fold generation logic based on whether synthetic data is used
    if config.get('use_synthetic_data', False):
        # if synthetic data is used, apply KFold cross-validation
        print("Using synthetic data: Applying KFold cross-validation with train, val, and test sets")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # generates train, validation, and test indices for KFold
        fold_gen = [
            (train_index[:int(0.8 * len(train_index))], train_index[int(0.8 * len(train_index)):], test_index)
            for train_index, test_index in kf.split(input_df)
        ]
    else:
        # for real data, using custom fold generator
        print("Using real data: Applying fold_generator_3_independent_indices")
        fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='survey', n_splits=5))
        # fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='unconditional', n_splits=5))

    # iterating over all folds and missing ratios to compute or load data
    for fold, (train_index, val_index, test_index) in enumerate(fold_gen):
        print(f"Processing fold {fold + 1} with masking={masking}")
        
        # file paths for train, validation and test sets for this fold
        fold_dir_all = os.path.join(fold_dir, f'fold_{fold}')
        os.makedirs(fold_dir_all, exist_ok=True)
        
        train_file = os.path.join(fold_dir_all, 'X_train_final.pkl')
        val_file = os.path.join(fold_dir_all, 'X_val_final.pkl')
        test_file = os.path.join(fold_dir_all, 'X_test_final.pkl')

        # checking if train, validation and test sets already exist, if not compute and save them
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            print(f"Loading precomputed train, val, and test sets for fold {fold + 1}")
            X_train_final = safe_load_pickle(train_file)
            X_val_final = safe_load_pickle(val_file)
            X_test_final = safe_load_pickle(test_file)
        else:
            print(f"Generating and saving train, val, and test sets for fold {fold + 1}")
            X_train_final, X_val_final, X_test_final = input_df.loc[train_index], input_df.loc[val_index], input_df.loc[test_index]
            save_pickle(train_file, X_train_final)
            save_pickle(val_file, X_val_final)
            save_pickle(test_file, X_test_final)
            
        for missing_ratio in missing_ratios:
            print(f"Processing missing ratio {missing_ratio}")

            # Define file paths for the current fold and missing ratio
            scaled_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_scaled_final_{missing_ratio}_fold_{fold}_repeat_0.pkl')
            mask_file = os.path.join(final_imputations_dir_missing_ratio, f'missing_mask_{missing_ratio}_fold_{fold}_repeat_0.pkl')
            train_time_files = {imputer: os.path.join(task3_time_dir, f"{imputer}_train_timing_{missing_ratio}_fold_{fold}_repeat_0.pkl") for imputer in imputers}
            test_time_files = {imputer: os.path.join(task3_time_dir, f"{imputer}_test_timing_{missing_ratio}_fold_{fold}_repeat_0.pkl") for imputer in imputers}
            imputed_data_files = {imputer: os.path.join(final_imputations_dir_missing_ratio, f'{imputer}_imputed_{missing_ratio}_fold_{fold}_repeat_0.pkl') for imputer in imputers}

            # ensuring all files for the current fold and missing ratio exist
            core_files_exist = os.path.exists(scaled_file) and os.path.exists(mask_file) and all(os.path.exists(file) for file in train_time_files.values()) and all(os.path.exists(file) for file in test_time_files.values()) and all(os.path.exists(file) for file in imputed_data_files.values())
            #and all(os.path.exists(file) for file in imputed_data_files.values()

            # only skip if the files for the current fold and missing ratio exist
            if core_files_exist:
                print(f"All necessary files exist for fold {fold + 1} and missing ratio {missing_ratio}. Skipping imputation.")
            else:
                # if any file is missing, load or compute data
                print(f"Computing data for fold {fold + 1} and missing ratio {missing_ratio}...")
                compute_data(final_imputations_dir_missing_ratio, missing_ratio, fold, 0, X_train_final, X_val_final, X_test_final, config, imputers, masking)

    print("All data has been computed and saved as pickles.")

# ---------------------------------------------------------
# Compute Data for Feature Evaluation, Task2
# ---------------------------------------------------------

def compute_data_for_feature_evaluation(final_imputations_dir_features, feature_intervals, fold, repeat, X_train_final, X_val_final, X_test_final, config, imputers, masking):
    """
    Check if the necessary data files (X_test_scaled_final, missing mask, imputed data) already exist. If not, compute them.
    This is the modified function for the feature-based evaluation.
    """
            
    standardized_imputers = {imputer: imputer.replace(' ', '_') for imputer in imputers}
    masking_dir, final_imputations_dir_missing_ratio, feature_eval_dir, task3_time_dir = get_final_imputations_dir(config)

    # RMSE store to collect results for different imputers across feature intervals
    rmse_store = {imputer: {n_features: [] for n_features in feature_intervals} for imputer in imputers}

    # iterating over the feature intervals (e.g., 15, 30, 45, etc.)
    for n_features in feature_intervals:
        print(f"Processing for {n_features} features...")

        # randomly select `n_features` features from the dataset
        selected_features = random.sample(list(X_train_final.columns), n_features)

        # using only the selected features for training, validation, and testing sets
        X_train_selected = X_train_final[selected_features]
        X_val_selected = X_val_final[selected_features]
        X_test_selected = X_test_final[selected_features]

        # defining file names based on the number of features, fold, and repeat
        feature_str = f"{n_features}_features_fold_{fold}_repeat_{repeat}"
        scaled_file = os.path.join(feature_eval_dir, f'X_test_scaled_final_{feature_str}.pkl')
        mask_file = os.path.join(feature_eval_dir, f'missing_mask_{feature_str}.pkl')
        imputed_data_files = {imputer: os.path.join(feature_eval_dir, f'{standardized_imputers[imputer]}_imputed_{feature_str}.pkl') for imputer in imputers}
        scaled_train_file = os.path.join(feature_eval_dir, f'X_train_scaled_{feature_str}.pkl')
        noisy_train_file = os.path.join(feature_eval_dir, f'X_train_scaled_noisy_{feature_str}.pkl')
        train_with_missing_file = os.path.join(feature_eval_dir, f'X_train_with_missing_{feature_str}.pkl')
        test_with_missing_file = os.path.join(feature_eval_dir, f'X_test_with_missing_{feature_str}.pkl')
        val_with_missing_file = os.path.join(feature_eval_dir, f'X_val_with_missing_{feature_str}.pkl')
        val_imputed_file = os.path.join(feature_eval_dir, f"X_val_knn_imputed_aes_{feature_str}.pkl")
        test_imputed_file = os.path.join(feature_eval_dir, f"X_test_knn_imputed_aes_{feature_str}.pkl")

        # if the necessary files do not exist, compute the required data
        print('Necessary files do not exist; they will be computed now from A to Z.')

        selected_option = config['process_nans']
        assert isinstance(selected_option, str), "Expected 'process_nans' to be a string"
    
        if not config.get('use_synthetic_data', False):
            if selected_option in ['numerical_only_drop_20_percentage_nans', 'keep_all_numerical']:
        
                X_train_selected = X_train_selected.select_dtypes(include=[np.number])
                X_val_selected = X_val_selected.select_dtypes(include=[np.number])
                X_test_selected = X_test_selected.select_dtypes(include=[np.number])
    
                # performing KNN imputation or load saved imputed data
                X_train_selected, X_val_selected, X_test_selected = first_knn_imp_for_numerical_data_with_missing_feature_evaluation(
                    X_train_selected, X_val_selected, X_test_selected, fold, masking_dir, n_features, repeat, config)

        # performing scaling on train, validation and test data
        if selected_option == 'drop_all_nans':
            X_train_selected = X_train_selected.select_dtypes(include=[np.number])
            X_val_selected = X_val_selected.select_dtypes(include=[np.number])
            X_test_selected = X_test_selected.select_dtypes(include=[np.number])
        
        X_train_scaled_final, X_val_scaled_final, X_test_scaled_final = prepare_data_for_feature_evaluation(X_train_selected, X_val_selected, X_test_selected, config)

        noise_factor = 0.2
        X_train_scaled_noisy = X_train_scaled_final + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_scaled_final.shape)
        X_train_scaled_noisy = np.clip(X_train_scaled_noisy, 0., 1.)  
        
        # saving the scaled X_train and X_test
        with open(scaled_train_file, 'wb') as f:
            pickle.dump(X_train_scaled_final, f)
        with open(scaled_file, 'wb') as f:
            pickle.dump(X_test_scaled_final, f)
        print(f"Scaled X_test saved for fold {fold}, repetition {repeat}, and {n_features} features.")

        # applying masking to test and validation data
        X_train_with_missing, _ = apply_masking(X_train_scaled_final.copy(), masking, 0.3)
        with open(train_with_missing_file, 'wb') as f:
            pickle.dump(X_train_with_missing, f)

        X_val_with_missing, _ = apply_masking(X_val_scaled_final.copy(), masking, 0.3)
        with open(val_with_missing_file, 'wb') as f:
            pickle.dump(X_val_with_missing, f)

        X_test_with_missing, missing_mask = apply_masking(X_test_scaled_final.copy(), masking, 0.3)
        with open(test_with_missing_file, 'wb') as f:
            pickle.dump(X_test_with_missing, f)

        with open(mask_file, 'wb') as f:
            pickle.dump(missing_mask, f)
        print(f"Missing mask saved for fold {fold}, repetition {repeat}, and {n_features} features.")

        # performing initial KNN imputation for AES
        X_val_imputed_final, X_test_imputed_final = initial_knn_imputed_data_for_aes_feature_evaluation(X_train_scaled_final, X_val_with_missing, X_test_with_missing, fold, masking_dir, n_features, repeat, config)
        with open(val_imputed_file, 'wb') as f:
            pickle.dump(X_val_imputed_final, f)
        with open(test_imputed_file, 'wb') as f:
            pickle.dump(X_test_imputed_final, f)

        imputed_data_store = {}
        # performing imputation for each imputer
        for imputer_name in imputers:
            print(f"Processing the {imputer_name} for fold {fold}, repetition {repeat}, and {n_features} features...")
            if imputer_name == 'AE':
                tf.keras.backend.clear_session()
                imputed_data = run_autoencoder_feature_evaluation(X_train_scaled_final, X_test_imputed_final, X_val_imputed_final, final_imputations_dir_features, fold, repeat, n_features)
            elif imputer_name == 'DAE':
                tf.keras.backend.clear_session()
                imputed_data = run_dae_feature_evaluation(X_train_scaled_noisy, X_test_imputed_final, X_val_imputed_final, final_imputations_dir_features, fold, repeat, n_features)
            elif imputer_name == 'VAE':
                tf.keras.backend.clear_session()
                imputed_data = run_vae_feature_evaluation(X_train_scaled_final, X_test_imputed_final, X_val_imputed_final, final_imputations_dir_features, fold, repeat, n_features)
            elif imputer_name == 'GAIN':
                tf.keras.backend.clear_session()
                imputed_data = run_gain_feature_evaluation(X_train_scaled_final, X_test_with_missing, X_val_with_missing, final_imputations_dir_features, fold, repeat, n_features)
            else:
                tf.keras.backend.clear_session()
                imputed_data = process_single_imputation_feature_evaluation(X_train_with_missing, X_test_scaled_final, X_test_with_missing, n_features, fold, repeat, final_imputations_dir_features).get(imputer_name)

            # saving the imputed data for future use
            imputed_data_store[imputer_name] = pd.DataFrame(imputed_data, columns=X_test_scaled_final.columns)
            with open(imputed_data_files[imputer_name], 'wb') as f:
                pickle.dump(imputed_data_store[imputer_name], f)
            print(f"Imputed data for {imputer_name} saved for fold {fold}, repetition {repeat}, and {n_features} features.")

        return X_test_scaled_final, missing_mask, imputed_data_store

# ---------------------------------------------------------
# Load or Compute Data Part for Feature Evaluation, Task2
# ---------------------------------------------------------

def load_or_compute_data_feature_evaluation(input_df, imputers, feature_intervals, config, masking):
    """
    Load or compute data based on the number of features and perform imputation.
    """
    if not config.get('use_synthetic_data', False):
        if 'process_nans' not in config or not isinstance(config['process_nans'], str):
            config = select_process_nans_option(config)
            print(f" The selected process_nans: {config['process_nans']}")

    # directories for storing results
    masking_dir, final_imputations_dir_missing_ratio, feature_eval_dir, task3_time_dir = get_final_imputations_dir(config)

    # paths for fold data
    fold_file_path = os.path.join(feature_eval_dir, "all_generator_folds")
    fold_dir = os.path.join(fold_file_path, "folds")
    os.makedirs(fold_file_path, exist_ok=True)
    os.makedirs(fold_dir, exist_ok=True)

    # generates 5-fold cross-validation splits
    if config.get('use_synthetic_data', False):
        print("Using synthetic data: Applying KFold cross-validation with train, val, and test sets")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_gen = [
            (train_index[:int(0.8 * len(train_index))], train_index[int(0.8 * len(train_index)):], test_index)
            for train_index, test_index in kf.split(input_df)
        ]
    else:
        print("Using real data: Applying fold_generator_3_independent_indices")
        fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='survey', n_splits=5))
        # fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='unconditional', n_splits=5))

    # iterates over all folds and feature intervals to compute or load data
    for fold, (train_index, val_index, test_index) in enumerate(fold_gen):
        print(f"Processing fold {fold + 1} with masking={masking}")

        # file paths for train, validation, and test sets for this fold
        fold_dir_all = os.path.join(fold_dir, f'fold_{fold}')
        os.makedirs(fold_dir_all, exist_ok=True)

        train_file = os.path.join(fold_dir_all, 'X_train_final.pkl')
        val_file = os.path.join(fold_dir_all, 'X_val_final.pkl')
        test_file = os.path.join(fold_dir_all, 'X_test_final.pkl')

        # loading or saving train, validation and test sets for the fold
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            print(f"Loading precomputed train, val, and test sets for fold {fold + 1}")
            X_train_final = safe_load_pickle(train_file)
            X_val_final = safe_load_pickle(val_file)
            X_test_final = safe_load_pickle(test_file)
        else:
            print(f"Generating and saving train, val, and test sets for fold {fold + 1}")
            X_train_final, X_val_final, X_test_final = input_df.loc[train_index], input_df.loc[val_index], input_df.loc[test_index]
            save_pickle(train_file, X_train_final)
            save_pickle(val_file, X_val_final)
            save_pickle(test_file, X_test_final)

        # iterating over different numbers of features to compute or load data
        for n_features in feature_intervals:
            print(f"Processing number of features {n_features}...")

            # defining file paths for the current fold and number of features
            feature_str = f"{n_features}_features_fold_{fold}_repeat_0"
            scaled_file = os.path.join(feature_eval_dir, f'X_test_scaled_final_{feature_str}.pkl')
            mask_file = os.path.join(feature_eval_dir, f'missing_mask_{feature_str}.pkl')
            imputed_data_files = {
                imputer: os.path.join(feature_eval_dir, f'{imputer}_imputed_{feature_str}.pkl') for imputer in imputers
            }

            # ensuring all files for the current fold and number of features exist
            core_files_exist = os.path.exists(scaled_file) and os.path.exists(mask_file) and all(os.path.exists(file) for file in imputed_data_files.values())

            # only skip if the files for the current fold and number of features exist
            if core_files_exist:
                print(f"All necessary files exist for fold {fold + 1} and number of features {n_features}. Skipping imputation.")
            else:
                # if any file is missing, load or compute data
                print(f"Computing data for fold {fold + 1} and number of features {n_features}...")
                compute_data_for_feature_evaluation(
                    feature_eval_dir, feature_intervals=[n_features], fold=fold, repeat=0,
                    X_train_final=X_train_final, X_val_final=X_val_final, X_test_final=X_test_final,
                    config=config, imputers=imputers, masking=masking
                )

    print("All data has been computed and saved as pickles.")