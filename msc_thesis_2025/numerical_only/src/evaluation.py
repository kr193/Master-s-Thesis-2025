#!/usr/bin/env python3
# File: src/evaluation.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pickle
import numpy as np
import pandas as pd
from src.utils import *
from src.visualization import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from src.preprocessing import fold_generator_3_independent_indices
from src.run_pipeline import compute_data, load_or_compute_data_part
from src.config_loader import get_final_imputations_dir
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------
# Evaluation Function for Task 1 & Task 3 
# ----------------------------------------

def evaluate_metrics_part(input_df, imputers, missing_ratios, config):
    """
    Evaluate metrics using precomputed and saved imputed data and masks.
    """
    masking_dir, final_imputations_dir_missing_ratio, feature_eval_dir, task3_time_dir = get_final_imputations_dir(config)

    # generating 5-fold cross-validation splits
    if config.get('use_synthetic_data', False):
        print("Using synthetic data: Applying KFold cross-validation with train, val, and test sets")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_gen = [
            (train_index[:int(0.8 * len(train_index))],
             train_index[int(0.8 * len(train_index)):],
             test_index)
            for train_index, test_index in kf.split(input_df)
        ]
    else:
        print("Using real data: Applying fold_generator_3_independent_indices")
        fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='survey', n_splits=5))
        # fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='unconditional', n_splits=5))

    # ensuring output_file_prefix exists
    output_file_prefix = config.get('output_file_prefix')
    if output_file_prefix is None:
        # fallback based on masking
        output_file_prefix = 'with_masking' if config.get('masking', False) else 'without_masking'
        config['output_file_prefix'] = output_file_prefix

    # initializing dictionaries to store metrics results
    rmse_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    nrmse_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    r2_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    mae_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    corr_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    train_time_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    test_time_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}

    extracted_values = {}  # Store actual and imputed values for each imputer and missing ratio

    for missing_ratio in missing_ratios:
        # iterating over all folds
        for fold, (train_index, val_index, test_index) in enumerate(fold_gen):
            print(f"Evaluating metrics for fold {fold + 1} and missing ratio {missing_ratio}")
            X_train_final, X_val_final, X_test_final = input_df.loc[train_index], input_df.loc[val_index], input_df.loc[test_index]

            # defining file paths
            scaled_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_scaled_final_{missing_ratio}_fold_{fold}_repeat_0.pkl')
            mask_file = os.path.join(final_imputations_dir_missing_ratio, f'missing_mask_{missing_ratio}_fold_{fold}_repeat_0.pkl')

            # loading the necessary data
            X_test_scaled_final = safe_load_pickle(scaled_file)
            missing_mask = safe_load_pickle(mask_file)

            if X_test_scaled_final is not None and missing_mask is not None:
                for imputer_name in imputers:
                    try:
                        # loading imputed data for the current imputer
                        # imputed_data = load_imputed_data(imputer_name, missing_ratio, fold)
                        imputed_data = load_imputed_data(imputer_name, missing_ratio, fold, config)

                        # print(f"imputed_data for {imputer_name}:", imputed_data )

                        if imputed_data is not None:
                            print(f"Imputed data shape for {imputer_name}: {imputed_data.shape}")  # Ensure this prints 
                            # checking if imputed data has only 1 column and needs to be reshaped
                        
                        # extracting actual and imputed values using mask
                        actual_values_extracted = pd.DataFrame(extract_values_using_mask(X_test_scaled_final, missing_mask))
                        imputed_df = pd.DataFrame(imputed_data)
                        imputed_df_extracted = pd.DataFrame(extract_values_using_mask(imputed_data, missing_mask))

                        print(f"Shape of actual_values_extracted: {actual_values_extracted.shape}")
                        print(f"Shape of imputed_df_extracted: {imputed_df_extracted.shape}")
                        
                        # storing the extracted values for scatter plotting for one fold only
                        if missing_ratio not in extracted_values:
                            extracted_values[missing_ratio] = {}
                        if imputer_name not in extracted_values[missing_ratio]:
                            extracted_values[missing_ratio][imputer_name] = {
                                'actual': actual_values_extracted,
                                'imputed': imputed_df_extracted
                            }
                            
                        # if fold == 3:
                        #     if missing_ratio not in extracted_values:
                        #         extracted_values[missing_ratio] = {}
                        #     if imputer_name not in extracted_values[missing_ratio]:
                        #         extracted_values[missing_ratio][imputer_name] = {
                        #             'actual': actual_values_extracted,
                        #             'imputed': imputed_df_extracted
                        #         }
  
                        # RMSE and other metrics calculations
                        rmse_value = np.sqrt(mean_squared_error(actual_values_extracted, imputed_df_extracted))
                        rmse_store[imputer_name][missing_ratio].append(rmse_value)

                        # NRMSE (Normalized RMSE)
                        nrmse_value = rmse_value / (actual_values_extracted.max() - actual_values_extracted.min())
                        nrmse_store[imputer_name][missing_ratio].append(nrmse_value)
    
                        # R²
                        r2_value = r2_score(actual_values_extracted, imputed_df_extracted)
                        r2_store[imputer_name][missing_ratio].append(r2_value)
    
                        # MAE (Mean Absolute Error)
                        mae_value = mean_absolute_error(actual_values_extracted, imputed_df_extracted)
                        mae_store[imputer_name][missing_ratio].append(mae_value)
    
                        # Correlation (Pearson)
                        corr_value, _ = pearsonr(actual_values_extracted.values.flatten(), imputed_df_extracted.values.flatten())
                        corr_store[imputer_name][missing_ratio].append(corr_value)
    
                        print(f"Appended RMSE value {rmse_value} and r2 score {r2_value} and correlation {corr_value} for imputer {imputer_name} and {missing_ratio} features.")

                    except ValueError as e:
                        print(f"Error for {imputer_name} at fold {fold + 1} and missing ratio {missing_ratio}: {e}")

                # now handle the time data separately, ensuring it doesn't interfere with the imputed data calculations
                for imputer_name in imputers:
                    training_time = load_time_data(imputer_name, missing_ratio, fold, config, time_type='train')
                    test_time = load_time_data(imputer_name, missing_ratio, fold, config, time_type='test')

                    print("training_time", training_time)
                    print("test_time", test_time)

                    # ensuring training and test time are loaded properly
                    if training_time is not None and test_time is not None:
                        train_time_store[imputer_name][missing_ratio].append(training_time)
                        test_time_store[imputer_name][missing_ratio].append(test_time)

    # returning both the final results and the time metrics
    # processing timing metrics
    time_metrics = {
        'imputer': [],
        'missing_ratio': [],
        'training_time': [],
        'test_time': []
    }

    for imputer in imputers:
        for ratio in missing_ratios:
            avg_train_time = np.mean(train_time_store[imputer][ratio]) if train_time_store[imputer][ratio] else None
            avg_test_time = np.mean(test_time_store[imputer][ratio]) if test_time_store[imputer][ratio] else None

            time_metrics['imputer'].append(imputer)
            time_metrics['missing_ratio'].append(ratio)
            time_metrics['training_time'].append(avg_train_time)
            time_metrics['test_time'].append(avg_test_time)

    final_results = {}
    for imputer in imputers:
        final_results[imputer] = {}
        for ratio in missing_ratios:
            rmse_vals = rmse_store[imputer][ratio]
            nrmse_vals = nrmse_store[imputer][ratio]
            r2_vals = r2_store[imputer][ratio]
            mae_vals = mae_store[imputer][ratio]
            corr_vals = corr_store[imputer][ratio]

            if rmse_vals:
                mean_rmse = np.mean(rmse_vals)
                std_rmse = np.std(rmse_vals)
                mean_nrmse = np.mean(nrmse_vals)
                std_nrmse = np.std(nrmse_vals)
                mean_r2 = np.mean(r2_vals)
                std_r2 = np.std(r2_vals)
                mean_mae = np.mean(mae_vals)
                std_mae = np.std(mae_vals)
                mean_corr = np.mean(corr_vals)
                std_corr = np.std(corr_vals)

                final_results[imputer][ratio] = {
                    'rmse': {'mean': mean_rmse, 'std': std_rmse},
                    'nrmse': {'mean': mean_nrmse, 'std': std_nrmse},
                    'r2': {'mean': mean_r2, 'std': std_r2},
                    'mae': {'mean': mean_mae, 'std': std_mae},
                    'corr': {'mean': mean_corr, 'std': std_corr}
                }
            else:
                final_results[imputer][ratio] = {
                    'rmse': {'mean': None, 'std': None},
                    'nrmse': {'mean': None, 'std': None},
                    'r2': {'mean': None, 'std': None},
                    'mae': {'mean': None, 'std': None},
                    'corr': {'mean': None, 'std': None}
                }

    # saving the results to Excel files
    save_metrics_to_excel(final_results, imputers, missing_ratios, config)
    return final_results,  extracted_values, time_metrics

# ---------------------------------------------------------
# Evaluation Function for Task 2 - Feature vs RMSE Analysis
# ---------------------------------------------------------

def evaluate_metrics_part_feature_evaluation(input_df, imputers, feature_intervals, config):
    """
    Evaluate metrics using precomputed and saved imputed data and masks, based on the number of features.

    Parameters:
    - input_df: The input dataset.
    - imputers: List of imputation methods to evaluate.
    - feature_intervals: List of feature intervals to evaluate.
    - config: Configuration settings.

    Returns:
    - final_results: Dictionary containing evaluation metrics (RMSE, NRMSE, R², MAE, Correlation) for each imputer and number of features.
    """
    masking_dir, final_imputations_dir_missing_ratio, feature_eval_dir, task3_time_dir = get_final_imputations_dir(config)
    
    # generating 5-fold cross-validation splits
    if config.get('use_synthetic_data', False):
        print("Using synthetic data: Applying KFold cross-validation with train, val, and test sets")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_gen = [
            (train_index[:int(0.8 * len(train_index))],
             train_index[int(0.8 * len(train_index)):],
             test_index)
            for train_index, test_index in kf.split(input_df)
        ]
    else:
        print("Using real data: Applying fold_generator_3_independent_indices")
        fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='survey', n_splits=5))
        # fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='unconditional', n_splits=5))
    
    # generating 5-fold cross-validation splits
    
    # initializing dictionaries to store metrics results
    rmse_store = {imputer: {n_features: [] for n_features in feature_intervals} for imputer in imputers}
    # nrmse_store = {imputer: {n_features: [] for n_features in feature_intervals} for imputer in imputers}
    r2_store = {imputer: {n_features: [] for n_features in feature_intervals} for imputer in imputers}
    mae_store = {imputer: {n_features: [] for n_features in feature_intervals} for imputer in imputers}
    corr_store = {imputer: {n_features: [] for n_features in feature_intervals} for imputer in imputers}

    # iterating over each number of features
    for n_features in feature_intervals:
        print(f"Evaluating metrics for {n_features} features")

        # iterating over all folds
        for fold, (train_index, val_index, test_index) in enumerate(fold_gen):
            print(f"Processing fold {fold + 1} for {n_features} features")

            X_train_final, X_val_final, X_test_final = input_df.loc[train_index], input_df.loc[val_index], input_df.loc[test_index]

            # defining file paths for scaled and masked data
            scaled_file = os.path.join(feature_eval_dir, f'X_test_scaled_final_{n_features}_features_fold_{fold}_repeat_0.pkl')
            mask_file = os.path.join(feature_eval_dir, f'missing_mask_{n_features}_features_fold_{fold}_repeat_0.pkl')

            # loading the necessary data
            X_test_scaled_final = safe_load_pickle(scaled_file)
            missing_mask = safe_load_pickle(mask_file)

            if X_test_scaled_final is None or missing_mask is None:
                print(f"Failed to load scaled data or missing mask for fold {fold + 1} and {n_features} features. Skipping...")
                continue

            # processing each imputer
            for imputer_name in imputers:
                try:
                    # loading imputed data for the current imputer and number of features
                    imputed_data = load_imputed_data_feature_evaluation(imputer_name, n_features, fold, config)

                    if imputed_data is None:
                        print(f"Imputed data missing for {imputer_name}. Skipping.")
                        continue

                    # extracting actual and imputed values using the mask
                    actual_values_extracted = extract_values_using_mask(X_test_scaled_final, missing_mask)
                    imputed_df_extracted = extract_values_using_mask(imputed_data, missing_mask)

                    if actual_values_extracted.empty or imputed_df_extracted.empty:
                        print(f"Empty data for imputer {imputer_name}, fold {fold + 1}, {n_features} features. Skipping...")
                        continue

                    # RMSE
                    rmse_value = np.sqrt(mean_squared_error(actual_values_extracted, imputed_df_extracted))
                    rmse_store[imputer_name][n_features].append(rmse_value)

                    # # NRMSE (Normalized RMSE)
                    # nrmse_value = rmse_value / (actual_values_extracted.max() - actual_values_extracted.min())
                    # nrmse_store[imputer_name][n_features].append(nrmse_value)

                    # R²
                    r2_value = r2_score(actual_values_extracted, imputed_df_extracted)
                    r2_store[imputer_name][n_features].append(r2_value)

                    # MAE (Mean Absolute Error)
                    mae_value = mean_absolute_error(actual_values_extracted, imputed_df_extracted)
                    mae_store[imputer_name][n_features].append(mae_value)

                    # Correlation (Pearson)
                    corr_value, _ = pearsonr(actual_values_extracted.values.flatten(), imputed_df_extracted.values.flatten())
                    corr_store[imputer_name][n_features].append(corr_value)

                    print(f"Appended RMSE value {rmse_value} and r2 score {r2_value} and correlation {corr_value} for imputer {imputer_name} and {n_features} features.")

                except ValueError as e:
                    print(f"Error for {imputer_name} at fold {fold + 1} and {n_features} features: {e}")

    # calculating final results
    final_results = {}
    for imputer in imputers:
        final_results[imputer] = {}
        for n_features in feature_intervals:
            rmse_vals = rmse_store[imputer][n_features]
            # nrmse_vals = nrmse_store[imputer][n_features]
            r2_vals = r2_store[imputer][n_features]
            mae_vals = mae_store[imputer][n_features]
            corr_vals = corr_store[imputer][n_features]

            if rmse_vals:
                mean_rmse = np.mean(rmse_vals)
                std_rmse = np.std(rmse_vals)
                # mean_nrmse = np.mean(nrmse_vals)
                # std_nrmse = np.std(nrmse_vals)
                mean_r2 = np.mean(r2_vals)
                std_r2 = np.std(r2_vals)
                mean_mae = np.mean(mae_vals)
                std_mae = np.std(mae_vals)
                mean_corr = np.mean(corr_vals)
                std_corr = np.std(corr_vals)

                final_results[imputer][n_features] = {
                    'rmse': {'mean': mean_rmse, 'std': std_rmse},
                    # 'nrmse': {'mean': mean_nrmse, 'std': std_nrmse},
                    'r2': {'mean': mean_r2, 'std': std_r2},
                    'mae': {'mean': mean_mae, 'std': std_mae},
                    'corr': {'mean': mean_corr, 'std': std_corr}
                }
            else:
                final_results[imputer][n_features] = {
                    'rmse': {'mean': None, 'std': None},
                    # 'nrmse': {'mean': None, 'std': None},
                    'r2': {'mean': None, 'std': None},
                    'mae': {'mean': None, 'std': None},
                    'corr': {'mean': None, 'std': None}
                }
    # saving the results to Excel files
    save_metrics_to_excel_feature_evaluation(final_results, imputers, feature_intervals, config)
    return final_results

# function to handle both masking and without masking
def handle_masking_and_evaluation(input_df, imputers, missing_ratios, config, masking):

    # loading or computing data and saving pickles
    load_or_compute_data_part(input_df, imputers, missing_ratios, config, config['masking'])
    
    # evaluating Metrics using Saved Pickles
    final_results, extracted_values, time_metrics = evaluate_metrics_part(input_df, imputers, missing_ratios, config)

    # plotting RMSE statistics
    missing_ratio_vs_stats(final_results, imputers, missing_ratios, config)

    # saving time metrics to Excel
    save_time_metrics_to_excel(time_metrics, config)

    # plotting time metrics (for visualizing training and test time per imputation method)
    plot_time_vs_missing_ratio(time_metrics, config)

    # creating scatter plots for the entire df
    combined_df_scatter_plots(extracted_values, config, missing_ratio=0.3)

    # creating scatter plots for the entire df
    per_column_scatter_plots(extracted_values, important_columns, imputers, config, missing_ratio=0.3)