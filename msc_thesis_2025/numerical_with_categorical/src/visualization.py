#!/usr/bin/env python3
# File: src/visualization.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import RESULTS_DIR

# RMSE vs Missing Ratio plotting and Accuracy vs Missing Ratio plotting
def missing_ratio_vs_stats(results, imputers, missing_ratios, output_file_prefix):
    """
    Visualization of Metric Trends Across Missing Ratios

    This function generates performance visualizations (RMSE & Accuracy) across missing ratios
    for various imputation methods. Plots include:
    
    1. RMSE vs Missing Ratio (mean values for numerical columns)
    2. Accuracy vs Missing Ratio (mean values for categorical columns)

    Parameters:
    - results: Dictionary containing evaluation metrics per imputer and missing ratio.
    - imputers: List of imputation methods evaluated.
    - missing_ratios: List of missing value percentages.
    - output_file_prefix: Prefix for naming output files.
    """
    print("Results keys:", results.keys())  # available keys in the results
    print("Expected imputers:", imputers)  # expected imputers list

    # adding jitter to 'Mean' imputer to reduce visual overlap
    results = add_jitter_to_mean_accuracy(results, imputers, missing_ratios, jitter_std=0.005)

    # initializing lists for RMSE statistics (numerical columns)
    all_mean_rmse = {} # Mean RMSE per imputer per missing ratio
    all_std_rmse_data = {imputer: [] for imputer in imputers}  # Std deviation for boxplot
    all_rmse_data = {imputer: [] for imputer in imputers}  # Raw RMSE values for boxplots

    # initializing lists for Accuracy statistics (categorical columns) 
    all_mean_accuracy = {}
    all_accuracy_data = {imputer: [] for imputer in imputers}  # Store Accuracy values for boxplot

    # ---------------------------------------------------------
    # iterating through all imputers and missing ratios to extract
    # RMSE (numerical) and Accuracy (categorical) statistics
    # ---------------------------------------------------------
    
    # collecting the RMSE and Accuracy mean and std data for each imputer and missing ratio
    for imputer in imputers:
        if imputer not in results:
            print(f"Warning: {imputer} not found in results. Skipping...")
            continue

        mean_rmse = []
        mean_accuracy = []
        
        for ratio in missing_ratios:
            # RMSE metrics
            rmse_values = results[imputer].get(ratio, {}).get('rmse', {})
            if rmse_values:
                mean_rmse_value = rmse_values['mean']
                std_rmse_value = rmse_values['std']
                all_values = rmse_values.get('values', [])  # list of all RMSE values (if available)

                mean_rmse.append(mean_rmse_value)
                all_std_rmse_data[imputer].append(std_rmse_value)
                all_rmse_data[imputer].extend(all_values)

            # Accuracy metrics
            accuracy_values = results[imputer].get(ratio, {}).get('accuracy', {})
            if accuracy_values:
                mean_accuracy_value = accuracy_values['mean']
                all_accuracy_values = accuracy_values.get('values', [])  # list of all Accuracy values 

                mean_accuracy.append(mean_accuracy_value)
                all_accuracy_data[imputer].extend(all_accuracy_values)

        # stores RMSE and Accuracy metrics for each imputer
        all_mean_rmse[imputer] = mean_rmse
        all_mean_accuracy[imputer] = mean_accuracy

    # -----------------------------------------------------
    # defining a color palette for imputers visualization
    # -----------------------------------------------------
    full_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#8c564b', # brown
        '#d62728',  # red
        '#9467bd',  # purple 
        '#e377c2',  # pink
    ]
    
    color_map = {imputer: full_colors[i % len(full_colors)] for i, imputer in enumerate(imputers)}

    # -------------------------------------------------------
    # Plot 1: RMSE vs Missing Ratio (Numerical)
    # -------------------------------------------------------
    # RMSE vs Missing Ratio Plot (Mean RMSE for Numerical Columns)
    plt.figure(figsize=(10, 6))
    
    for imputer in imputers:
        if imputer in all_mean_rmse and all_mean_rmse[imputer]:  # ensuring data exists
            rmse_values = np.nan_to_num(all_mean_rmse[imputer])  # replacing NaNs with zeros if necessary
            
            # ensures valid data before plotting
            if len(rmse_values) == len(missing_ratios):
                plt.plot(
                    missing_ratios,
                    rmse_values,
                    label=f'{imputer} RMSE (mean)',
                    marker='o',
                    color=color_map[imputer]
                )
            else:
                print(f"Skipping {imputer} in RMSE plot due to missing data.")
    
    plt.title('RMSE vs Missing Ratio Across Imputers (Mean)')
    plt.xlabel('Missing Ratio')
    plt.ylabel('Mean RMSE')
    
    # dynamically adjust y-axis to fit the range of RMSE values
    if all_mean_rmse:
        min_rmse = min([min(vals) for vals in all_mean_rmse.values() if len(vals) > 0])
        max_rmse = max([max(vals) for vals in all_mean_rmse.values() if len(vals) > 0])
        plt.ylim(min_rmse * 0.9, max_rmse * 1.1)  # Adding buffer to limits
    
    plt.legend()
    plt.grid(True)
    rmse_vs_missing_filename = os.path.join(RESULTS_DIR, f'{output_file_prefix}_rmse_vs_missing_ratio.png')
    plt.savefig(rmse_vs_missing_filename)
    plt.show()
    print(f'✅ RMSE vs Missing Ratio plot saved to {rmse_vs_missing_filename}')


    # ---------------------------------------------------------
    # Plot 2: Accuracy vs Missing Ratio (Categorical)
    # ---------------------------------------------------------
    ### Accuracy vs Missing Ratio Plot (Mean Accuracy for Categorical Columns)
    plt.figure(figsize=(10, 6))
    
    for imputer in imputers:
        if imputer in all_mean_accuracy and all_mean_accuracy[imputer]:  # ensuring data exists
            acc_values = np.nan_to_num(all_mean_accuracy[imputer])  # replacing NaNs with zeros
    
            if len(acc_values) == len(missing_ratios):  # ensuring data consistency
                plt.plot(
                    missing_ratios,
                    acc_values, 
                    label=f'{imputer} Accuracy (mean)', 
                    marker='o', 
                    color=color_map[imputer]
                )
            else:
                print(f" Skipping {imputer} in Accuracy plot due to missing data.")
    
    plt.title('Accuracy vs Missing Ratio Across Imputers (Mean)')
    plt.xlabel('Missing Ratio')
    plt.ylabel('Mean Accuracy')
    
    # focused Y-axis range from 0.5 to 0.8 
    plt.ylim(0.5, 1)
    
    plt.legend()
    plt.grid(True)
    accuracy_vs_missing_filename = os.path.join(RESULTS_DIR, f'{output_file_prefix}_accuracy_vs_missing_ratio.png')
    plt.savefig(accuracy_vs_missing_filename)
    plt.show()
    print(f'Accuracy vs Missing Ratio plot saved to {accuracy_vs_missing_filename}')
    
def add_jitter_to_mean_accuracy(results, imputers, missing_ratios, jitter_std=0.002):
    """
    Add slight vertical jitter (Gaussian noise) to 'Mean' method accuracy values to avoid overlapping with others.
    
    Parameters:
    - results: The dictionary storing evaluation results.
    - imputers: List of imputer method names.
    - missing_ratios: List of missing data ratios.
    - jitter_std: Standard deviation of the Gaussian noise to apply.
    
    Returns:
    - Modified results dictionary with jittered accuracy values for 'Mean' imputer.
    """
    if 'Mean' not in imputers:
        print("'Mean' imputer not found in imputers list.")
        return results

    for ratio in missing_ratios:
        accuracy_data = results.get('Mean', {}).get(ratio, {}).get('accuracy', {})
        if accuracy_data and 'mean' in accuracy_data:
            jitter = np.random.normal(0, jitter_std)
            original = accuracy_data['mean']
            results['Mean'][ratio]['accuracy']['mean'] += jitter
            print(f"Added jitter {jitter:.4f} to 'Mean' accuracy at missing ratio {ratio:.2f} (was {original:.4f})")

    return results