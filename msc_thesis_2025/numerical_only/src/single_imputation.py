#!/usr/bin/env python3
# File: src/single_imputation.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import time
import pickle
import numpy as np
import pandas as pd
from src.utils import *
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from src.config_loader import get_final_imputations_dir
from sklearn.experimental import enable_iterative_imputer
from src.config_loader import load_config, get_final_imputations_dir
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# -----------------------------------------------------------------------------------------------
# Non-deep learning (statistical and traditional ML) methods for aggregated numerical DHS dataset
# -----------------------------------------------------------------------------------------------

def process_single_imputation(train_data_with_missing, test_data, test_data_with_missing, missing_ratio, fold, repeat, output_dir, config):
    """
    To perform or load imputed data using different imputation methods.
    
    - This function applies various imputation methods (Mean, KNN and MICE) to handle missing data. 
    - It saves and loads imputed data, statistical and machine learning method and corresponding timing information by avoiding redundant computations.
    
    Parameters:
    - train_data_with_missing: Training data with missing values.
    - test_data: Original test data without missing values.
    - test_data_with_missing: Test data with missing values.
    - missing_ratio: different missing ratios from 10% to 60%
    - fold: Fold number.
    - repeat: Repetition number, which can be increased for robust evaluation. Currently, it is set to 1 for time consumption.
    - output_dir: Directory for saving/loading results.
    - config: specific configuration set by user

    Returns:
    - imputed_data: Dictionary containing the imputed data for each imputer.

    """
    masking_dir, final_imputations_dir_missing_ratio, task2_dir, task3_time_dir = get_final_imputations_dir(config)
    
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'KNN': KNNImputer(n_neighbors=5),
        # 'MICE_Bayesian': IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=6688)
        'MICE_Ridge': IterativeImputer(estimator=Ridge(), max_iter=1, random_state=6688)
    }

    imputed_data = {}
    # initializing dictionaries
    training_time_data = {imputer: {} for imputer in imputers}
    test_time_data = {imputer: {} for imputer in imputers}

    for imputer_name, imputer in imputers.items():
        imputed_data_path = os.path.join(output_dir, f"{imputer_name}_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
        imputer_model_path = os.path.join(output_dir, f"{imputer_name}_trained_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
        train_timing_info_path = os.path.join(task3_time_dir, f"{imputer_name}_train_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
        test_timing_info_path = os.path.join(task3_time_dir, f"{imputer_name}_test_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")

        # checking if the imputed data and timing files exist
        if os.path.exists(imputed_data_path) and os.path.exists(imputer_model_path):
            # loading imputed data and timing info
            imputed_df = pd.read_pickle(imputed_data_path)
            train_timing_info = pd.read_pickle(train_timing_info_path)
            test_timing_info = pd.read_pickle(test_timing_info_path)
            print(f"imputed data {imputed_df.shape} for {imputer_name} are loaded from disk successfully")
            print(f"no need for re-computation")
            
            if imputed_df is not None and imputed_df.isnull().sum().sum() == 0:
                # data is valid
                print(f"{imputer_name} loaded from disk with shape: {imputed_df.shape}")
                imputed_data[imputer_name] = imputed_df
                training_time_data[imputer_name] = train_timing_info
                test_time_data[imputer_name] = test_timing_info
                # continue  
        else: 
            
            # if imputed data does not exist or is invalid, recompute the imputation
            print(f"As there is no pickle files saved before, Fitting {imputer_name} for fold {fold}, repetition {repeat}...")
    
            # tracking training time
            start_train_time = time.time()
            # Fit the imputer on the training data with missing values
            imputer.fit(train_data_with_missing)
            training_time = time.time() - start_train_time
    
            # imputing the test data
            start_test_time = time.time()
            imputed_df = pd.DataFrame(imputer.transform(test_data_with_missing), columns=test_data_with_missing.columns)
            test_time = time.time() - start_test_time
    
            # checking if the imputation was successful (no NaNs and non-empty DataFrame)
            if imputed_df.isnull().sum().sum() == 0 and len(imputed_df) > 0:
                print(f"Imputation successful for {imputer_name}. Saving files...")
                save_pickle(imputed_data_path, imputed_df)
                save_pickle(imputer_model_path, imputer)
                save_pickle(train_timing_info_path, training_time)
                save_pickle(test_timing_info_path, test_time)
            else:
                print(f"Imputation failed for {imputer_name}. Empty DataFrame or NaN values detected.")
                raise ValueError(f"Imputation failed for {imputer_name}. NaN values or empty DataFrame detected.")
    
            # storing the newly imputed data and timings
            imputed_data[imputer_name] = imputed_df
            training_time_data[imputer_name] = training_time
            test_time_data[imputer_name] = test_time
    
            print(f"Training time data: {training_time_data}")
            print(f"Test time data: {test_time_data}")
    
            # calculating and printing RMSE for validation
            rmse = np.sqrt(mean_squared_error(test_data, imputed_df))
            print(f"RMSE for {imputer_name}: {rmse}")

    return imputed_data, training_time_data, test_time_data

# -----------------------------------------------------------------------------------------------------------------------
# Non-deep learning (statistical and traditional ML) methods for aggregated numerical DHS dataset for features evaluation
# -----------------------------------------------------------------------------------------------------------------------

def process_single_imputation_feature_evaluation(train_data_with_missing, test_data, test_data_with_missing, n_features, fold, repeat, output_dir):
    """
    Perform or load imputed data using different imputation methods based on the number of selected features.
    
    Parameters:
    - train_data_with_missing: Training data with missing values.
    - test_data: Original test data without missing values.
    - test_data_with_missing: Test data with missing values.
    - n_features: Number of features being used in this evaluation.
    - fold: Fold number.
    - repeat: Repetition number, which can be increased for robust evaluation. Currently, it is set to 1 for time consumption.
    - output_dir: Directory for saving/loading results.

    Returns:
    - imputed_data: Dictionary containing the imputed data for each imputer.
    """
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'KNN': KNNImputer(n_neighbors=5),
        'MICE_Ridge': IterativeImputer(estimator=Ridge(), max_iter=1, random_state=6688)
    }

    imputed_data = {}

    for imputer_name, imputer in imputers.items():
        # constructing file names based on number of features
        imputed_data_path = os.path.join(output_dir, f"{imputer_name}_imputed_{n_features}_features_fold_{fold}_repeat_{repeat}.pkl")
        imputer_model_path = os.path.join(output_dir, f"{imputer_name}_trained_{n_features}_features_fold_{fold}_repeat_{repeat}.pkl")

        # loading imputed data if available
        imputed_df = safe_load_pickle(imputed_data_path)

        # checking if both imputed data and imputer model already exist and are valid
        if os.path.exists(imputed_data_path) and os.path.exists(imputer_model_path):
            print(f"{imputer_name} loaded from disk for {n_features} features with shape: {imputed_df.shape}")
            imputed_df = pd.read_pickle(imputed_data_path)

            # checking if the loaded imputed data contains NaNs and recompute if so
            if imputed_df is not None and imputed_df.isnull().sum().sum() == 0:
                imputed_data[imputer_name] = imputed_df
                print(f"Successfully loaded valid imputed data for {imputer_name} for {n_features} features.")
                rmse = np.sqrt(mean_squared_error(test_data, imputed_df))
                print(f"RMSE for {imputer_name} with {n_features} features: {rmse}")
            else:
                print(f"Invalid imputed data found for {imputer_name} with {n_features} features. Recomputing.")
        else:
            # fitting the imputer on the cleaned training data
            print(f"Fitting {imputer_name} for {n_features} features, fold {fold}, repetition {repeat}...")
            imputer.fit(train_data_with_missing)
    
            # imputing the test data
            imputed_df = pd.DataFrame(imputer.transform(test_data_with_missing),
                                      columns=test_data_with_missing.columns)
    
            # checking for NaN values after imputation
            if imputed_df.isnull().sum().sum() == 0:
                print(f"Imputation successful for {imputer_name} with {n_features} features. No NaN values detected.")
            else:
                print(f"Imputation failed for {imputer_name} with {n_features} features. NaN values detected.")
    
            # saving imputed data and model
            with open(imputed_data_path, 'wb') as f:
                pickle.dump(imputed_df, f)
            with open(imputer_model_path, 'wb') as f:
                pickle.dump(imputer, f)
            print(f"Imputed data and model saved for {imputer_name} with {n_features} features.")

            imputed_data[imputer_name] = imputed_df
    
        # calculating RMSE for validation
        rmse = np.sqrt(mean_squared_error(test_data, imputed_df))
        print(f"RMSE for {imputer_name} with {n_features} features: {rmse}")

    return imputed_data