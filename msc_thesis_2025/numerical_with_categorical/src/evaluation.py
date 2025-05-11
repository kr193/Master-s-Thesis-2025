#!/usr/bin/env python3
# File: src/evaluation.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.visualization import missing_ratio_vs_stats
from src.config import RESULTS_DIR, GET_FINAL_IMPUTATIONS_DIR
from src.run_pipeline import compute_data, load_or_compute_data_part
from src.masking import extract_values_using_mask, apply_masking_for_cat_num
from src.preprocessing import prepare_data, fold_generator_3_independent_indices
from src.utils import decode_one_hot_cached, convert_sparse_to_dense, safe_load_pickle, load_imputed_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# loading pickles for each (fold, ratio), calculates RMSE and appends to list
def evaluate_metrics_part(input_df, imputers, missing_ratios, config):
    """
    Evaluate performance metrics (RMSE, MAE, R2, correlation, accuracy, precision, recall, F1)
    for each imputation method across all missing ratios and folds.

    Parameters:
    - input_df: Original input DataFrame.
    - imputers: List of imputer method names.
    - missing_ratios: List of missing ratios to evaluate.
    - config: Dictionary containing configuration options.

    Returns:
    - final_results: Dictionary with mean and std of metrics for each imputer and missing ratio.
    - extracted_values: Dictionary storing actual and imputed values used for scatter plots.
    """
    # defining necessary directories
    final_imputations_dir_missing_ratio = GET_FINAL_IMPUTATIONS_DIR(config)

    # generating stratified folds using a custom split strategy for real survey data
    print("Using real data: Applying fold_generator_3_independent_indices")
    fold_gen = list(fold_generator_3_independent_indices(input_df, split_type='survey', n_splits=5))

    # initializing dictionaries to store metrics results
    rmse_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    nrmse_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    r2_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    mae_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    corr_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}

    # initializing dictionaries to store categorical metrics
    accuracy_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    precision_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    recall_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}
    f1_store = {imputer: {ratio: [] for ratio in missing_ratios} for imputer in imputers}

    extracted_values = {}  # storing actual and imputed values for each imputer and missing ratio

    # ---------------------------------------------------------
    # Main Evaluation Loop
    # ---------------------------------------------------------
    for missing_ratio in missing_ratios:
        # iterating over all folds
        for fold, (train_index, val_index, test_index) in enumerate(fold_gen):
            print(f"Evaluating metrics for fold {fold + 1} and missing ratio {missing_ratio}")

            # Loading fold-specific datasets
            X_train_final, X_val_final, X_test_final = input_df.loc[train_index], input_df.loc[val_index], input_df.loc[test_index]

            # Define file paths
            encoder_file = os.path.join(final_imputations_dir_missing_ratio, f'encoder_fold_{fold}_repeat_0.pkl')
            scaled_file = os.path.join(final_imputations_dir_missing_ratio, f'X_test_scaled_final_{missing_ratio}_fold_{fold}_repeat_0.pkl')
            mask_file = os.path.join(final_imputations_dir_missing_ratio, f'missing_mask_{missing_ratio}_fold_{fold}_repeat_0.pkl')

            # Load the necessary data
            X_test_scaled_final = safe_load_pickle(scaled_file)
            missing_mask = safe_load_pickle(mask_file)
            encoder = safe_load_pickle(encoder_file)

            if X_test_scaled_final is not None and missing_mask is not None:
                for imputer_name in imputers:
                    try:
                        # loading imputed data for the current imputer
                        imputed_data = load_imputed_data(imputer_name, missing_ratio, fold, config)
                        # print(f"imputed_data for {imputer_name}:", imputed_data )

                        # print("Before sparse-to-dense conversion:", imputed_data.shape)
                        if isinstance(imputed_data, pd.DataFrame):
                            imputed_data = convert_sparse_to_dense(imputed_data)
                        # print("After sparse-to-dense conversion:", imputed_data.shape)

                        if imputed_data is not None:
                            # from numpy to df
                            imputed_df = pd.DataFrame(imputed_data)
                            # before one-hot encoding (with original categorical columns)
                            numerical_columns = X_test_scaled_final.select_dtypes(exclude=['object', 'category']).columns
                            categorical_columns = X_test_scaled_final.select_dtypes(include=['object', 'category']).columns

                            # decoding One-Hot encoded Data
                            decoded_df = decode_one_hot_cached(
                                            imputed_df,
                                            encoder,
                                            categorical_columns,
                                            fold,
                                            imputer_name,
                                            missing_ratio,
                                            cache_dir=os.path.join(final_imputations_dir_missing_ratio, "decode_cache"),
                                            binary_threshold=2,
                                            binarize_threshold=0.6
                                        )
                            # print(decoded_df.shape) 

                            # ensuring decoded categorical data has the same columns as the original X_test
                            missing_columns = set(X_test_scaled_final.columns) - set(decoded_df.columns)
                            if missing_columns:
                                print(f" Warning: Missing columns after decoding: {missing_columns}")
                                # ensuring all expected columns exist post-decoding
                                for col in missing_columns:
                                    decoded_df[col] = X_test_scaled_final[col]  # restoring missing numerical columns

                            # print("Decoded X_test shape:", X_test_scaled_final.shape)
                            # print("Decoded Imputed shape:", decoded_df.shape)

                            # ensuring values are comparable for accuracy
                            decoded_df.fillna("missing", inplace=True)
                            X_test_scaled_final.fillna("missing", inplace=True)

                            # computing accuracy or row-wise accuracy across all columns
                            accuracy = (X_test_scaled_final.values == decoded_df.values).mean()
                            # print(f'Accuracy for {imputer_name} at {missing_ratio} missingness: {accuracy}')
                            accuracy_store[imputer_name][missing_ratio].append(accuracy)
                            imputed_df = decoded_df  # assigning back after decoding

                        # extracting actual and imputed values using mask
                        actual_values_extracted = pd.DataFrame(extract_values_using_mask(X_test_scaled_final, missing_mask))
                        imputed_df_extracted = pd.DataFrame(extract_values_using_mask(imputed_df, missing_mask))
                        # print(f"Shape of actual_values_extracted: {actual_values_extracted.shape}")
                        # print(f"Shape of imputed_df_extracted: {imputed_df_extracted.shape}")

                        # separate actual and imputed data into numerical/categorical
                        actual_numerical = actual_values_extracted[numerical_columns]
                        imputed_numerical = imputed_df_extracted[numerical_columns]
                        
                        actual_categorical = actual_values_extracted[categorical_columns]
                        imputed_categorical = imputed_df_extracted[categorical_columns]

                        # storing the extracted values for scatter plotting
                        if missing_ratio not in extracted_values:
                            extracted_values[missing_ratio] = {}
                        # saving extracted values for visualizations (scatter plots)
                        if imputer_name not in extracted_values[missing_ratio]:
                            extracted_values[missing_ratio][imputer_name] = {
                                'actual': actual_values_extracted,
                                'imputed': imputed_df_extracted
                            }

                        # metrics calculations for numerical columns
                        # numeric metrics
                        if not actual_numerical.empty:
                            rmse_value = np.sqrt(mean_squared_error(actual_numerical, imputed_numerical))
                            # print(f'rmse value for {imputer_name}', rmse_value)
                            nrmse_value = rmse_value / (actual_numerical.max() - actual_numerical.min()).mean()
                            r2_value = r2_score(actual_numerical, imputed_numerical)
                            mae_value = mean_absolute_error(actual_numerical, imputed_numerical)
                            # corr_value = np.corrcoef(actual_numerical.T, imputed_numerical.T)[0, 1]
                            corr_value, _ = pearsonr(actual_numerical.values.flatten(), imputed_numerical.values.flatten())

                            # rmse_store is a nested dictionary 
                            # rmse_store[imputer][ratio] gives you a list of 5 RMSEs, one for each fold for that imputer and missing ratio.
                            rmse_store[imputer_name][missing_ratio].append(rmse_value) # a list of RMSEs per fold. This is per fold.
                                        # rmse_store = {
                                        #     'KNN': {
                                        #         0.1: [0.24, 0.25, 0.27, 0.26, 0.23],  # ← values across 5 folds for 10% missing
                                        #         0.2: [ ... ],
                                        #         ...
                                        #     },
                                        #     'Mean': {
                                        #         0.1: [ ... ],
                                        #         ...
                                        #     }
                                        # }
                            nrmse_store[imputer_name][missing_ratio].append(nrmse_value)
                            r2_store[imputer_name][missing_ratio].append(r2_value)
                            mae_store[imputer_name][missing_ratio].append(mae_value)
                            corr_store[imputer_name][missing_ratio].append(corr_value)

                        # ---------------------------------------------------------
                        # Evaluating Categorical Performance
                        # ---------------------------------------------------------
                        
                        # handling multi-class categorical comparison (single label per row)
                        # categorical metrics

                        # ensuring both actual and imputed categorical data are available
                        if not actual_categorical.empty and not imputed_categorical.empty:
                            accuracy_values = []
                            # computing accuracy for each categorical column separately
                            for col in categorical_columns:
                                # converting values to string for comparison (in case of mixed types)
                                acc = accuracy_score(actual_categorical[col].astype(str), imputed_categorical[col].astype(str))
                                accuracy_values.append(acc)

                            # mean accuracy across all categorical columns
                            mean_accuracy = np.mean(accuracy_values)  # computing mean accuracy across categorical columns
                            accuracy_store[imputer_name][missing_ratio].append(mean_accuracy)

                            # -------------------------------------
                            # Flatten categorical rows for precision/recall/F1
                            # Each row is converted into a string sequence ("0110")
                            # This is useful for row-wise accuracy or sequence matching
                            # -------------------------------------
                            actual_categorical_flat = actual_categorical.astype(str).agg(''.join, axis=1)
                            imputed_categorical_flat = imputed_categorical.astype(str).agg(''.join, axis=1)

                            precision_values = []
                            recall_values = []
                            f1_values = []

                            # -------------------------------------
                            # Column-wise precision, recall, and F1-score
                            # -------------------------------------
                            for col in categorical_columns:
                                # ---------------------------------------------------------
                                # here, average='macro':
                                # Treats each class equally, regardless of class imbalance.
                                # Especially suitable for multi-class categorical variables where all classes are equally important.
                                #
                                # zero_division=0:
                                # Avoids runtime warnings or NaNs when a class has no predicted or true samples
                                # This is especially helpful for sparse categories or small test sets.
                                # ---------------------------------------------------------
                                precision = precision_score(actual_categorical[col], imputed_categorical[col], average='macro', zero_division=0)
                                recall = recall_score(actual_categorical[col], imputed_categorical[col], average='macro', zero_division=0)
                                f1 = f1_score(actual_categorical[col], imputed_categorical[col], average='macro', zero_division=0)
                                # accumulating or average across columns
                                precision_values.append(precision)
                                recall_values.append(recall)
                                f1_values.append(f1)

                            # mean scores across all categorical columns
                            mean_precision = np.mean(precision_values)
                            mean_recall = np.mean(recall_values)
                            mean_f1 = np.mean(f1_values)

                            # storing the final averaged scores in the respective metrics dictionary
                            precision_store[imputer_name][missing_ratio].append(mean_precision)
                            recall_store[imputer_name][missing_ratio].append(mean_recall)
                            f1_store[imputer_name][missing_ratio].append(mean_f1)

                    # catching and report exceptions during metric computation
                    except ValueError as e:
                        print(f"Error for {imputer_name} at fold {fold + 1} and missing ratio {missing_ratio}: {e}")

    # ---------------------------------------------------------
    # Compute Mean and Std for Each Metric Output
    # ---------------------------------------------------------

    # dictionary to store the final aggregated results for each imputer and missing ratio.
    # each entry includes mean and standard deviation for all evaluation metrics.
    final_results = {} # stores mean ± std of all RMSEs for a given imputer and missing_ratio
                    # final_results = {
                    #   'KNN': {
                    #     0.1: {
                    #       'rmse': { 'mean': 0.25, 'std': 0.015 },
                    #       'mae': {...},
                    #       ...
                    #     },
                    #     0.2: {...},
                    #     ...
                    #   },
                    #   'Mean': {
                    #     0.1: { ... },
                    #     ...
                    #   }
                    # }

    # iterating through each imputation method
    for imputer in imputers:
        final_results[imputer] = {}
        
        # iterating through each missing ratio (10%, 20%, ...)
        for ratio in missing_ratios:
            # retrieving lists of metric values across all folds for the current imputer and missing ratio
            rmse_vals = rmse_store[imputer][ratio] # rmse_vals = [0.24, 0.25, 0.27, 0.26, 0.23]  # across 5 folds
            nrmse_vals = nrmse_store[imputer][ratio]
            r2_vals = r2_store[imputer][ratio]
            mae_vals = mae_store[imputer][ratio]
            corr_vals = corr_store[imputer][ratio]

            accuracy_vals = accuracy_store[imputer][ratio]
            precision_vals = precision_store[imputer][ratio]
            recall_vals = recall_store[imputer][ratio]
            f1_vals = f1_store[imputer][ratio]

            # if metric values exist (the imputer successfully processed this fold-ratio combo)
            if rmse_vals:
                # calculating mean and standard deviation for each numerical metric
                mean_rmse = np.mean(rmse_vals) # divides by 5, Done separately for each missing_ratio, for each imputer, 5 folds' values to get mean ± std  # → averages all 5 RMSEs for KNN @ 10% missingness, Mean RMSE = Average of [fold0, fold1, fold2, fold3, fold4]
                std_rmse = np.std(rmse_vals) # std RMSE = Standard deviation of those 5 values
                mean_nrmse = np.mean(nrmse_vals) # Normalized RMSE
                std_nrmse = np.std(nrmse_vals)
                mean_r2 = np.mean(r2_vals) # R² (Coefficient of Determination)
                std_r2 = np.std(r2_vals)
                mean_mae = np.mean(mae_vals) # Mean Absolute Error
                std_mae = np.std(mae_vals)
                mean_corr = np.mean(corr_vals) # Pearson correlation coefficient
                std_corr = np.std(corr_vals)

                # computing mean and std for classification-based metrics
                mean_accuracy = np.mean(accuracy_vals)
                std_accuracy = np.std(accuracy_vals)
                mean_precision = np.mean(precision_vals)
                std_precision = np.std(precision_vals)
                mean_recall = np.mean(recall_vals)
                std_recall = np.std(recall_vals)
                mean_f1 = np.mean(f1_vals)
                std_f1 = np.std(f1_vals)

                # storing all metrics in the final results dictionary for this imputer and ratio
                final_results[imputer][ratio] = {
                    'rmse': {'mean': mean_rmse, 'std': std_rmse},
                    'nrmse': {'mean': mean_nrmse, 'std': std_nrmse},
                    'r2': {'mean': mean_r2, 'std': std_r2},
                    'mae': {'mean': mean_mae, 'std': std_mae},
                    'corr': {'mean': mean_corr, 'std': std_corr},
                    'accuracy': {'mean': mean_accuracy, 'std': std_accuracy},
                    'precision': {'mean': mean_precision, 'std': std_precision},
                    'recall': {'mean': mean_recall, 'std': std_recall},
                    'f1': {'mean': mean_f1, 'std': std_f1}
                }
            else:
                # In case no metric values were computed (due to imputer failure)
                # Fill with None to indicate missing evaluation
                final_results[imputer][ratio] = {
                    'rmse': {'mean': None, 'std': None},
                    'nrmse': {'mean': None, 'std': None},
                    'r2': {'mean': None, 'std': None},
                    'mae': {'mean': None, 'std': None},
                    'corr': {'mean': None, 'std': None},
                    'accuracy': {'mean': None, 'std': None},
                    'precision': {'mean': None, 'std': None},
                    'recall': {'mean': None, 'std': None},
                    'f1': {'mean': None, 'std': None}
                }

    # saving the aggregated evaluation results to Excel
    save_metrics_to_excel(final_results, imputers, missing_ratios, config['output_file_prefix'])
    # returning the aggregated results and raw extracted values 
    return final_results, extracted_values

# =============================================================
# Function: calculate_metrics
# Purpose: to compute standard regression metrics for imputed data
# =============================================================
def calculate_metrics(y_true, y_pred):
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # R-squared score (coefficient of determination)
    r2 = r2_score(y_true, y_pred)
    # Pearson correlation coefficient
    corr, _ = pearsonr(y_true, y_pred)
    return mse, rmse, r2, corr

# ==================================================================================
# Function: save_metrics_to_excel
# Purpose: saving all evaluation metrics (mean ± std) to an Excel report
# ==================================================================================
def save_metrics_to_excel(final_results, imputers, missing_ratios, output_file_prefix):
    """
    Save RMSE, NRMSE, R², MAE, Correlation, Accuracy, Precision, Recall and F1-score statistics into individual and combined Excel files.

    Parameters:
    - final_results: The final results containing mean and std for each metric.
    - imputers: List of imputers evaluated.
    - missing_ratios: List of missing ratios evaluated.
    - output_file_prefix: The prefix for naming the output files.
    """
    try:
        # creating empty DataFrames for each metric
        rmse_df = pd.DataFrame(index=imputers)
        nrmse_df = pd.DataFrame(index=imputers)
        r2_df = pd.DataFrame(index=imputers)
        mae_df = pd.DataFrame(index=imputers)
        corr_df = pd.DataFrame(index=imputers)
        
        accuracy_df = pd.DataFrame(index=imputers)
        precision_df = pd.DataFrame(index=imputers)
        recall_df = pd.DataFrame(index=imputers)
        f1_df = pd.DataFrame(index=imputers)

        combined_df = pd.DataFrame(index=imputers)

        # filling each metric DataFrame for each missing ratio
        for ratio in missing_ratios:
            # numerical metrics
            rmse_df[f'RMSE {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['rmse']['mean']:.4f} ± {final_results[imputer][ratio]['rmse']['std']:.4f}"
                for imputer in imputers
            ]
            nrmse_df[f'NRMSE {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['nrmse']['mean']:.4f} ± {final_results[imputer][ratio]['nrmse']['std']:.4f}"
                for imputer in imputers
            ]
            r2_df[f'R² {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['r2']['mean']:.4f} ± {final_results[imputer][ratio]['r2']['std']:.4f}"
                for imputer in imputers
            ]
            mae_df[f'MAE {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['mae']['mean']:.4f} ± {final_results[imputer][ratio]['mae']['std']:.4f}"
                for imputer in imputers
            ]
            corr_df[f'Correlation {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['corr']['mean']:.4f} ± {final_results[imputer][ratio]['corr']['std']:.4f}"
                for imputer in imputers
            ]
            # categorical metrics
            accuracy_df[f'Accuracy {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['accuracy']['mean']:.4f} ± {final_results[imputer][ratio]['accuracy']['std']:.4f}"
                for imputer in imputers
            ]
            precision_df[f'Precision {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['precision']['mean']:.4f} ± {final_results[imputer][ratio]['precision']['std']:.4f}"
                for imputer in imputers
            ]
            recall_df[f'Recall {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['recall']['mean']:.4f} ± {final_results[imputer][ratio]['recall']['std']:.4f}"
                for imputer in imputers
            ]
            f1_df[f'F1-Score {ratio} (mean ± std)'] = [
                f"{final_results[imputer][ratio]['f1']['mean']:.4f} ± {final_results[imputer][ratio]['f1']['std']:.4f}"
                for imputer in imputers
            ]

            # adding all metrics into one combined DataFrame
            combined_df[f'RMSE {ratio} (mean ± std)'] = rmse_df[f'RMSE {ratio} (mean ± std)']
            combined_df[f'NRMSE {ratio} (mean ± std)'] = nrmse_df[f'NRMSE {ratio} (mean ± std)']
            combined_df[f'R² {ratio} (mean ± std)'] = r2_df[f'R² {ratio} (mean ± std)']
            combined_df[f'MAE {ratio} (mean ± std)'] = mae_df[f'MAE {ratio} (mean ± std)']
            combined_df[f'Correlation {ratio} (mean ± std)'] = corr_df[f'Correlation {ratio} (mean ± std)']
            combined_df[f'Accuracy {ratio} (mean ± std)'] = accuracy_df[f'Accuracy {ratio} (mean ± std)']
            combined_df[f'Precision {ratio} (mean ± std)'] = precision_df[f'Precision {ratio} (mean ± std)']
            combined_df[f'Recall {ratio} (mean ± std)'] = recall_df[f'Recall {ratio} (mean ± std)']
            combined_df[f'F1-Score {ratio} (mean ± std)'] = f1_df[f'F1-Score {ratio} (mean ± std)']

        # defining the output filename
        # combined_filename  = f'{output_file_prefix}_all_metrics_stats.xlsx'
        combined_filename = os.path.join(RESULTS_DIR, f'{output_file_prefix}_all_metrics_stats.xlsx')

        # saving all metric tables to an Excel w
        with pd.ExcelWriter(combined_filename) as writer:
            combined_df.to_excel(writer, index_label='Imputer')
            rmse_df.to_excel(writer, sheet_name='RMSE', index_label='Imputer')
            nrmse_df.to_excel(writer, sheet_name='NRMSE', index_label='Imputer')
            r2_df.to_excel(writer, sheet_name='R²', index_label='Imputer')
            mae_df.to_excel(writer, sheet_name='MAE', index_label='Imputer')
            corr_df.to_excel(writer, sheet_name='Correlation', index_label='Imputer')
            accuracy_df.to_excel(writer, sheet_name='Accuracy', index_label='Imputer')
            precision_df.to_excel(writer, sheet_name='Precision', index_label='Imputer')
            recall_df.to_excel(writer, sheet_name='Recall', index_label='Imputer')
            f1_df.to_excel(writer, sheet_name='F1-Score', index_label='Imputer')

        print(f'All metrics combined saved to {combined_filename}')
    except Exception as e:
        # handles and reports any errors during file creation
        print(f"Error saving Excel files: {e}")

# function to handle both masking and without masking scenarios
def handle_masking_and_evaluation(input_df, imputers, missing_ratios, config, masking):
    """
    Handles data processing, evaluation and plotting based on masking configuration.

    Parameters:
    - input_df: Original dataset before introducing missing values.
    - imputers: List of imputation methods to be evaluated.
    - missing_ratios: List of missing ratios to simulate.
    - config: Configuration dictionary.
    - masking: Boolean indicating if MAR/MCAR masking is applied.
    """

    # step 1: loading or computing data and saving pickles
    load_or_compute_data_part(input_df, imputers, missing_ratios, config, masking)
    
    # step 2: evaluating metrics using saved pickles on disk
    final_results, extracted_values = evaluate_metrics_part(input_df, imputers, missing_ratios, config)

    # step 3️: visualizing RMSE and accuracy across missing ratios
    missing_ratio_vs_stats(final_results, imputers, missing_ratios, output_file_prefix=config['output_file_prefix'])