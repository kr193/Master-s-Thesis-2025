#!/usr/bin/env python3
# File: src/visualization.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import string
import numpy as np
import pandas as pd
from src.utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.helper_functions import select_process_nans_option

# Task 1: saving metrics for Task1 - RMSE vs Missing Ratios
def save_metrics_to_excel(final_results, imputers, missing_ratios, config):
    """
    Save RMSE, NRMSE, R², MAE, and Correlation statistics into individual and combined Excel files.
    
    Parameters:
    - final_results: The final results containing mean and std for each metric.
    - imputers: List of imputers evaluated.
    - missing_ratios: List of missing ratios evaluated.
    - output_file_prefix: The prefix for naming the output files.
    """
    output_file_prefix = config['output_file_prefix']
    output_images_dir = config['output_images_dir']
    
    try:
        # empty dfs for each metric
        rmse_df = pd.DataFrame(index=imputers)
        nrmse_df = pd.DataFrame(index=imputers)
        r2_df = pd.DataFrame(index=imputers)
        mae_df = pd.DataFrame(index=imputers)
        corr_df = pd.DataFrame(index=imputers)
        combined_df = pd.DataFrame(index=imputers)

        # filling dfs with the results
        for ratio in missing_ratios:
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

            # combining all metrics into one DataFrame
            combined_df[f'RMSE {ratio} (mean ± std)'] = rmse_df[f'RMSE {ratio} (mean ± std)']
            combined_df[f'NRMSE {ratio} (mean ± std)'] = nrmse_df[f'NRMSE {ratio} (mean ± std)']
            combined_df[f'R² {ratio} (mean ± std)'] = r2_df[f'R² {ratio} (mean ± std)']
            combined_df[f'MAE {ratio} (mean ± std)'] = mae_df[f'MAE {ratio} (mean ± std)']
            combined_df[f'Correlation {ratio} (mean ± std)'] = corr_df[f'Correlation {ratio} (mean ± std)']

        # the output filename
        prefix = f"synthetic_{config['correlation_type']}" if config.get('use_synthetic_data', False) else config['process_nans']
        combined_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_all_metrics_stats.xlsx")

        # saving the dfs to Excel
        with pd.ExcelWriter(combined_filename) as writer:
            combined_df.to_excel(writer, index_label='Imputer')
            rmse_df.to_excel(writer, sheet_name='RMSE', index_label='Imputer')
            nrmse_df.to_excel(writer, sheet_name='NRMSE', index_label='Imputer')
            r2_df.to_excel(writer, sheet_name='R²', index_label='Imputer')
            mae_df.to_excel(writer, sheet_name='MAE', index_label='Imputer')
            corr_df.to_excel(writer, sheet_name='Correlation', index_label='Imputer')

        print(f'All metrics combined saved to {combined_filename}')
    except Exception as e:
        print(f"Error saving Excel files: {e}")

# Task 1: RMSE vs Missing Ratio plotting
def missing_ratio_vs_stats(results, imputers, missing_ratios, config):
    """
    Plot RMSE vs Missing Ratio, RMSE std boxplots, and RMSE values across methods.
    The function generates and saves three plots:
    1. RMSE vs Missing Ratio (mean values).
    2. RMSE Std Boxplots across imputers.
    3. RMSE Boxplots across imputers for all missing ratios.
    """
    output_file_prefix = config['output_file_prefix']
    output_images_dir = config['output_images_dir']
    
    # print("Results keys:", results.keys())  # available keys in the results
    # print("Expected imputers:", imputers)  # expected imputers list
    
    # initializing lists for RMSE statistics
    all_mean_rmse = {}
    all_std_rmse_data = {imputer: [] for imputer in imputers}  # storing RMSE std values for boxplot
    all_rmse_data = {imputer: [] for imputer in imputers}  # storing RMSE values for each imputer

    # collecting the RMSE mean and std data for each imputer and missing ratio
    for imputer in imputers:
        if imputer not in results:
            print(f"Warning: {imputer} not found in results. Skipping...")
            continue

        mean_rmse = []
        for ratio in missing_ratios:
            # accessing 'mean' and 'std' for RMSE
            rmse_values = results[imputer].get(ratio, {}).get('rmse', {})
            if not rmse_values:
                print(f"Warning: No RMSE data found for {imputer} at missing ratio {ratio}. Skipping...")
                continue

            mean_rmse_value = rmse_values['mean']
            std_rmse_value = rmse_values['std']
            all_values = rmse_values.get('values', [])  # list of all RMSE values 

            mean_rmse.append(mean_rmse_value)
            all_std_rmse_data[imputer].append(std_rmse_value)  # storing for std boxplot
            all_rmse_data[imputer].extend(all_values)  # collecting all RMSE values for the imputer

        # storing RMSE metrics for each imputer
        all_mean_rmse[imputer] = mean_rmse

    # defining a color palette for imputers visualization
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
    # RMSE vs Missing Ratio Plot (Mean RMSE) 
    plt.figure(figsize=(10, 8))

    # plotting RMSE vs Missing Ratio for each imputer with assigned colors
    for imputer in imputers:
        if imputer in all_mean_rmse:
            plt.plot(
                missing_ratios,
                all_mean_rmse[imputer],
                label=f'{imputer} RMSE (mean)',
                marker='o',
                color=color_map[imputer]  
            )

    # finalizing RMSE vs Missing Ratio plot
    plt.title(f'RMSE vs Missing Ratio Across Imputers (Mean)')
    plt.xlabel('Missing Ratio')
    plt.ylabel('Mean RMSE')
    plt.legend()
    plt.grid(True)

    # adjusts filename prefix based on synthetic or real data
    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']
    
    rmse_vs_missing_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_rmse_vs_missing_ratio_new.png")
    plt.savefig(rmse_vs_missing_filename)
    plt.show()

    print(f'RMSE vs Missing Ratio plot saved to {rmse_vs_missing_filename}')

    # initializing lists for RMSE statistics
    rmse_across_ratios = {imputer: [] for imputer in imputers}  # correctly initializing rmse_across_ratios dictionary
    all_std_rmse_data = {imputer: [] for imputer in imputers}  # storing RMSE std values for boxplot

    # collecting the RMSE mean and std data for each imputer and missing ratio
    for imputer in imputers:
        if imputer not in results:
            print(f"Warning: {imputer} not found in results. Skipping...")
            continue

        for ratio in missing_ratios:
            rmse_values = results[imputer].get(ratio, {}).get('rmse', {})
            if not rmse_values:
                print(f"Warning: No RMSE data found for {imputer} at missing ratio {ratio}. Skipping...")
                continue

            mean_rmse_value = rmse_values['mean']
            std_rmse_value = rmse_values['std']

            if mean_rmse_value is not None:
                rmse_across_ratios[imputer].append(mean_rmse_value)  # storing mean RMSE for this ratio
            if std_rmse_value is not None:
                all_std_rmse_data[imputer].append(std_rmse_value)  # storing std RMSE for this ratio

    # RMSE Boxplot Across Imputers (Mean RMSE for All Missing Ratios) 
    plt.figure(figsize=(10, 8))

    matte_colors = [
    '#FFCC80',  # Matte gray
    '#A3C1AD',  # Matte green
    '#B39DDB',  # Matte peach 
    '#F5A9BC',  # Matte pink
    '#9FA8DA',  # Matte blue
    '#FFE5CC', # Matte light orange
    '#C5E1A5',  # Matte light green
    '#90A4AE',  # Matte blue-gray
    ]

    # custom boxplot with matte colors and thicker mean line
    boxplot_data = [rmse_across_ratios[imputer] for imputer in imputers if rmse_across_ratios[imputer]]
    box = plt.boxplot(boxplot_data, patch_artist=True, showmeans=True, 
                      meanline=True, meanprops=dict(linestyle='-', linewidth=2, color='#8B0000'),
                      labels=[imputer for imputer in imputers if rmse_across_ratios[imputer]])

    # applying distinct matte colors to each box
    for patch, color in zip(box['boxes'], matte_colors):
        patch.set_facecolor(color)

    # setting title, labels and grid
    plt.title(f'RMSE Boxplot Across Imputers (Mean RMSE for All Missing Ratios)')
    plt.xlabel('Imputers')
    plt.ylabel('Mean RMSE')
    plt.grid(True)

    # saving and showing the plot
    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']

    rmse_boxplot_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_rmse_boxplot_across_methods.png")
    plt.tight_layout()
    plt.savefig(rmse_boxplot_filename, bbox_inches='tight')
    plt.show()

    print(f'RMSE Boxplot Across Methods saved to {rmse_boxplot_filename}')

    # RMSE Std Boxplot 
    plt.figure(figsize=(10, 8))

    # custom boxplot for RMSE standard deviations with matte colors and thicker mean line
    boxplot_data_std = [all_std_rmse_data[imputer] for imputer in imputers if all_std_rmse_data[imputer]]
    box_std = plt.boxplot(boxplot_data_std, patch_artist=True, showmeans=True,
                          meanline=True, meanprops=dict(linestyle='-', linewidth=2, color='#8B0000'),
                          labels=[imputer for imputer in imputers if all_std_rmse_data[imputer]])

    # applying distinct matte colors to each box
    for patch, color in zip(box_std['boxes'], matte_colors):
        patch.set_facecolor(color)

    # setting title, labels and grid
    plt.title(f'RMSE Std Boxplot Across Imputers for Different Missing Ratios')
    plt.xlabel('Imputers')
    plt.ylabel('RMSE Std Dev')
    plt.grid(True)

    # saving and showing the plot
    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']

    rmse_std_boxplot_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_rmse_std_boxplot_new.png")
    plt.tight_layout()
    plt.savefig(rmse_std_boxplot_filename, bbox_inches='tight')
    plt.show()

    print(f'RMSE Std Boxplot saved to {rmse_std_boxplot_filename}')

# Task 1: Plotting combined df's scatter plots for Task1 - RMSE vs Missing Ratio 
def combined_df_scatter_plots(extracted_values, config, missing_ratio=0.3):
    """
    Plots scatter plots with regression lines for the extracted actual and imputed values for the given missing ratio.
    """
    output_file_prefix = config['output_file_prefix']
    
    if missing_ratio not in extracted_values:
        print(f"No data available for missing ratio {missing_ratio}")
        return

    # defining colors consistent with plot_actual_vs_imputed_per_column
    scatter_color = '#00008B'  # dark blue for imputed values
    actual_line_color = '#ff7f0e'  # orange for actual values line

    # extracting the actual and imputed values for the given missing ratio
    imputed_data = extracted_values[missing_ratio]

    # a figure with a grid of 2 rows, 4 columns 
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharey=True)
    fig.suptitle(f'Actual vs Imputed Values (Missing Ratio = {missing_ratio})', fontsize=21)

    # flattening the axes and keep only the first 7
    axes = axes.flatten()

    # iterating over imputers and plot the scatter plot with regression line
    for idx, (imputer_name, data) in enumerate(imputed_data.items()):
        actual_values = data['actual'].values.flatten()
        imputed_values = data['imputed'].values.flatten()

        # calculating regression metrics
        mse, rmse, r2, corr = calculate_metrics(actual_values, imputed_values)

        # plotting the actual vs imputed values and the regression line
        axes[idx].scatter(actual_values, imputed_values, 
                          label=f'{imputer_name}\nR2: {r2:.3f}, RMSE: {rmse:.3f}, Corr: {corr:.3f}', 
                          alpha=0.6, color=scatter_color, marker='o')
        axes[idx].scatter(actual_values, actual_values, color=actual_line_color, alpha=0.6, label='Actual Values', marker='o')

        axes[idx].set_xlabel('Actual Values', fontsize=14)
        axes[idx].set_ylabel('Imputed Values', fontsize=14)

        # adding the legend and format it
        legend = axes[idx].legend(loc='upper left', fontsize=12)
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_linewidth(1.5)

    # hiding the last (8th) subplot
    fig.delaxes(axes[7])

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjusting layout to fit the title
    output_images_dir = config['output_images_dir']
    
    prefix = f"synthetic_{config['correlation_type']}" if config.get('use_synthetic_data', False) else config['process_nans']
    scatter_plot_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_scatter_plots_missing_ratio_{missing_ratio}.png")

    plt.savefig(scatter_plot_filename)
    plt.show()

    print(f'Scatter plot saved to {scatter_plot_filename}')

# helper function to sanitize file names
def sanitize_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned_filename = ''.join(c for c in filename if c in valid_chars)
    return cleaned_filename.replace(' ', '_')

# Task 1: per feature evaluation plotting function
def plot_actual_vs_imputed_per_column(col, actual_values, imputed_values_dict, save_path):
    """
    Plot actual vs imputed values for a given column across all imputers.
    """
    n_imputers = len(imputed_values_dict)
    colors = plt.cm.get_cmap('tab10', n_imputers).colors  
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharey=True)

    fig.suptitle(f'Actual vs Imputed Values for {col}', fontsize=21)

    for ax, (imputer_name, imputed_values), color in zip(axes.flat, imputed_values_dict.items(), colors):
        mse, rmse, r2, corr = calculate_metrics(actual_values, imputed_values)
        ax.scatter(actual_values, imputed_values, label=f'{imputer_name}\nR2: {r2:.3f}, RMSE: {rmse:.3f}, Corr: {corr:.3f}', alpha=0.6, color='#00008B', marker='o')
        ax.scatter(actual_values, actual_values, color='#ff7f0e', alpha=0.6, label='Actual Values', marker='o')
        ax.set_xlabel('Actual Values', fontsize=14)
        ax.set_ylabel('Imputed Values', fontsize=14)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True)) 
        legend = ax.legend(loc='upper left', fontsize=12)
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.show()

# Task 1: per feature evaluation with scattter plots in terms of RMSE vs Missing Ratio 
def per_column_scatter_plots(extracted_values, important_columns, imputers, config, missing_ratio=0.3):
    """
    Plots scatter plots with regression lines for the extracted actual and imputed values for the given missing ratio.
    This includes a combined plot for the whole dataset and individual plots for important columns.
    """
    # skipping plotting for synthetic data
    if config.get('use_synthetic_data', False):
        print("Skipping per-column scatter plots for synthetic data.")
        return
    
    output_file_prefix = config['output_file_prefix']
    
    if missing_ratio not in extracted_values:
        print(f"No data available for missing ratio {missing_ratio}")
        return

    # plotting scatter plots for the important columns
    for col in important_columns:
        actual_values_col = extracted_values[missing_ratio][imputers[0]]['actual'][col].values  
        imputed_values_dict = {
            imputer_name: extracted_values[missing_ratio][imputer_name]['imputed'][col].values
            for imputer_name in imputers
        }
        
        sanitized_col_name = sanitize_filename(col)
        output_images_dir = config['output_images_dir']
        save_path = os.path.join(output_images_dir, f"{config['process_nans']}_{output_file_prefix}_Actual_Imputed_{sanitized_col_name}.png")
        plot_actual_vs_imputed_per_column(col, actual_values_col, imputed_values_dict, save_path)
        

# ---------------------------- Task 2: RMSE vs Number of Features -----------------------------------

# Task 2: Saving metrics for Task 2 - RMSE vs Number of Features 
def save_metrics_to_excel_feature_evaluation(final_results, imputers, feature_intervals, config):
    """ 
    Save RMSE, NRMSE, R², MAE, and Correlation statistics into individual and combined Excel files.
    
    Parameters:
    - final_results: The final results containing mean and std for each metric.
    - imputers: List of imputers evaluated.
    - feature_intervals: List of feature intervals evaluated.
    - output_file_prefix: The prefix for naming the output files.
    """
    output_file_prefix = config['output_file_prefix']
    output_images_dir = config['output_images_dir']
        
    try:
        # empty dfs for each metric
        rmse_df = pd.DataFrame(index=imputers)
        # nrmse_df = pd.DataFrame(index=imputers)
        r2_df = pd.DataFrame(index=imputers)
        mae_df = pd.DataFrame(index=imputers)
        corr_df = pd.DataFrame(index=imputers)
        combined_df = pd.DataFrame(index=imputers)

        # filling dfs with the results
        for n_features in feature_intervals:
            rmse_df[f'RMSE {n_features} features (mean ± std)'] = [
                f"{final_results[imputer][n_features]['rmse']['mean']:.4f} ± {final_results[imputer][n_features]['rmse']['std']:.4f}"
                for imputer in imputers
            ]
            # nrmse_df[f'NRMSE {n_features} features (mean ± std)'] = [
            #     f"{final_results[imputer][n_features]['nrmse']['mean']:.4f} ± {final_results[imputer][n_features]['nrmse']['std']:.4f}"
            #     for imputer in imputers
            # ]
            r2_df[f'R² {n_features} features (mean ± std)'] = [
                f"{final_results[imputer][n_features]['r2']['mean']:.4f} ± {final_results[imputer][n_features]['r2']['std']:.4f}"
                for imputer in imputers
            ]
            mae_df[f'MAE {n_features} features (mean ± std)'] = [
                f"{final_results[imputer][n_features]['mae']['mean']:.4f} ± {final_results[imputer][n_features]['mae']['std']:.4f}"
                for imputer in imputers
            ]
            corr_df[f'Correlation {n_features} features (mean ± std)'] = [
                f"{final_results[imputer][n_features]['corr']['mean']:.4f} ± {final_results[imputer][n_features]['corr']['std']:.4f}"
                for imputer in imputers
            ]

            # combine all metrics into one DataFrame
            combined_df[f'RMSE {n_features} features (mean ± std)'] = rmse_df[f'RMSE {n_features} features (mean ± std)']
            # combined_df[f'NRMSE {n_features} features (mean ± std)'] = nrmse_df[f'NRMSE {n_features} features (mean ± std)']
            combined_df[f'R² {n_features} features (mean ± std)'] = r2_df[f'R² {n_features} features (mean ± std)']
            combined_df[f'MAE {n_features} features (mean ± std)'] = mae_df[f'MAE {n_features} features (mean ± std)']
            combined_df[f'Correlation {n_features} features (mean ± std)'] = corr_df[f'Correlation {n_features} features (mean ± std)']

        # the output filename
        prefix = f"synthetic_{config['correlation_type']}" if config.get('use_synthetic_data', False) else config['process_nans']
        combined_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_feature_evaluation_all_metrics_stats.xlsx")

        # saving the DataFrames to Excel
        with pd.ExcelWriter(combined_filename) as writer:
            combined_df.to_excel(writer, index_label='Imputer')
            rmse_df.to_excel(writer, sheet_name='RMSE', index_label='Imputer')
            # nrmse_df.to_excel(writer, sheet_name='NRMSE', index_label='Imputer')
            r2_df.to_excel(writer, sheet_name='R²', index_label='Imputer')
            mae_df.to_excel(writer, sheet_name='MAE', index_label='Imputer')
            corr_df.to_excel(writer, sheet_name='Correlation', index_label='Imputer')

        print(f'All metrics combined saved to {combined_filename}')
    except Exception as e:
        print(f"Error saving Excel files: {e}")

# Task 2: RMSE vs Number of Features plotting
def num_of_features_vs_stats(results, imputers, feature_intervals, config):
    """
    Plot RMSE vs Number of Features, RMSE std boxplots, and RMSE values across methods.
    The function generates and saves three plots:
    1. RMSE vs Number of Features (mean values).
    2. RMSE Std Boxplots across imputers.
    3. RMSE Boxplots across imputers for all feature intervals.
    """
    output_file_prefix = config['output_file_prefix']
    output_images_dir = config['output_images_dir']

    print("Results keys:", results.keys())  # available keys in the results
    print("Expected imputers:", imputers)  # expected imputers list

    # initializing lists for RMSE statistics
    all_mean_rmse = {}
    rmse_across_features = {imputer: [] for imputer in imputers}  # correctly initializing rmse_across_features dictionary
    all_std_rmse_data = {imputer: [] for imputer in imputers}  # storing RMSE std values for boxplot

    # collecting the RMSE mean and std data for each imputer and feature interval
    for imputer in imputers:
        if imputer not in results:
            print(f"Warning: {imputer} not found in results. Skipping...")
            continue

        mean_rmse = []
        for n_features in feature_intervals:
            #'mean' and 'std' for RMSE
            rmse_values = results[imputer].get(n_features, {}).get('rmse', {})
            if not rmse_values:
                print(f"Warning: No RMSE data found for {imputer} at {n_features} features. Skipping...")
                continue

            mean_rmse_value = rmse_values['mean']
            std_rmse_value = rmse_values['std']

            if mean_rmse_value is not None:
                mean_rmse.append(mean_rmse_value)  # storing mean RMSE for this feature interval
                rmse_across_features[imputer].append(mean_rmse_value)  # storing mean RMSE for this feature interval
            if std_rmse_value is not None:
                all_std_rmse_data[imputer].append(std_rmse_value)  # storing std RMSE for this feature interval

        # storing RMSE metrics for each imputer
        all_mean_rmse[imputer] = mean_rmse

    # a list of solid colors for the plot 
    colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#8c564b',
        '#d62728', 
        '#9467bd', 
        '#e377c2'
    ]

    color_map = {imputer: colors[i % len(colors)] for i, imputer in enumerate(imputers)}

    # RMSE vs Number of Features Plot (Mean RMSE)
    plt.figure(figsize=(10, 8))

    # plotting RMSE vs Number of Features for each imputer with assigned colors
    for imputer in imputers:
        if imputer in all_mean_rmse:
            plt.plot(
                feature_intervals,
                all_mean_rmse[imputer],
                label=f'{imputer} RMSE (mean)',
                marker='o',
                color=color_map[imputer]  
            )

    # finalizing RMSE vs Number of Features plot
    plt.title(f'RMSE vs Number of Features Across Imputers (Mean)')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean RMSE')
    plt.legend()
    plt.grid(True)

    # adjusting filename prefix based on synthetic or real data
    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']
    
    rmse_vs_features_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_rmse_vs_num_features.png")
    plt.savefig(rmse_vs_features_filename)
    plt.show()

    print(f'RMSE vs Number of Features plot saved to {rmse_vs_features_filename}')

    # RMSE Boxplot Across Imputers (Mean RMSE for All Feature Intervals) 
    plt.figure(figsize=(10, 8))

    # custom boxplot with colors and thicker mean line
    boxplot_data = [rmse_across_features[imputer] for imputer in imputers if rmse_across_features[imputer]]
    box = plt.boxplot(boxplot_data, patch_artist=True, showmeans=True, 
                      meanline=True, meanprops=dict(linestyle='-', linewidth=2, color='firebrick'),
                      labels=[imputer for imputer in imputers if rmse_across_features[imputer]])

    # applying distinct colors to each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # adding a legend
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors[:len(imputers)]], 
               imputers, loc='upper right', title="Imputers")

    # setting title, labels and grid
    plt.title(f'RMSE Boxplot Across Imputers (Mean RMSE for All Feature Intervals)')
    plt.xlabel('Imputers')
    plt.ylabel('Mean RMSE')
    plt.grid(True)

    # saving and showing the plot
    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']
    
    rmse_boxplot_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_rmse_boxplot_num_features.png")
    plt.savefig(rmse_boxplot_filename)
    plt.show()

    print(f'RMSE Boxplot Across Methods saved to {rmse_boxplot_filename}')

    # RMSE Std Boxplot 
    plt.figure(figsize=(10, 8))

    # custom boxplot for RMSE standard deviations with colors and a thicker mean line
    boxplot_data_std = [all_std_rmse_data[imputer] for imputer in imputers if all_std_rmse_data[imputer]]
    box_std = plt.boxplot(boxplot_data_std, patch_artist=True, showmeans=True,
                          meanline=True, meanprops=dict(linestyle='-', linewidth=2, color='firebrick'),
                          labels=[imputer for imputer in imputers if all_std_rmse_data[imputer]])

    # applying distinct colors to each box
    for patch, color in zip(box_std['boxes'], colors):
        patch.set_facecolor(color)

    # setting title, labels and grid
    plt.title(f'RMSE Std Boxplot Across Imputers for Different Numbers of Features')
    plt.xlabel('Imputers')
    plt.ylabel('RMSE Std Dev')
    plt.grid(True)

    # saving and showing the plot
    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']
    
    rmse_std_boxplot_filename = os.path.join(output_images_dir, f"{prefix}_{output_file_prefix}_rmse_std_boxplot_num_features.png")
    plt.savefig(rmse_std_boxplot_filename)
    plt.show()

    print(f'RMSE Std Boxplot saved to {rmse_std_boxplot_filename}')

# Task 3: RMSE vs Time plotting
def plot_time_vs_missing_ratio(time_metrics, config):
    """
    Plot average training and test times vs missing ratios for different imputation methods.
    """
    output_file_prefix = config['output_file_prefix']
    output_images_dir = config['output_images_dir']
    
    df = pd.DataFrame(time_metrics)
    
    # colors to be used for different imputation methods
    full_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#8c564b',  # brown
        '#d62728',  # red
        '#9467bd',  # purple
        '#e377c2',  # pink
    ]
    
    # a color map to assign each imputer a specific color
    imputer_color_map = {imputer: full_colors[i % len(full_colors)] for i, imputer in enumerate(df['imputer'].unique())}

    # determining prefix based on synthetic or real data
    if config.get('use_synthetic_data', False):
        prefix = f"synthetic_{config['correlation_type']}"
    else:
        prefix = config['process_nans']

    #### --- Training time plot --- ####
    plt.figure(figsize=(10, 6))
    for imputer in df['imputer'].unique():
        imputer_data = df[df['imputer'] == imputer]
        plt.plot(imputer_data['missing_ratio'], imputer_data['training_time'], label=f'{imputer}', marker='o', color=imputer_color_map[imputer])

    plt.title('Average Training Time vs Missing Ratio')
    plt.xlabel('Missing Ratio')
    plt.ylabel('Training Time (seconds)')
    plt.legend()
    plt.grid(True)

    # training plot filename
    train_time_filename = f"{prefix}_{output_file_prefix}_average_training_time_vs_missing_ratio.png"
    train_time_path = os.path.join(output_images_dir, train_time_filename)
    print(f"Saving training time plot to: {train_time_path}")
    plt.savefig(train_time_path)
    plt.show()  

    #### --- Test time plot --- ####
    plt.figure(figsize=(10, 6))
    for imputer in df['imputer'].unique():
        imputer_data = df[df['imputer'] == imputer]
        plt.plot(imputer_data['missing_ratio'], imputer_data['test_time'], label=f'{imputer}', marker='o', color=imputer_color_map[imputer])

    plt.title('Average Test Time vs Missing Ratio')
    plt.xlabel('Missing Ratio')
    plt.ylabel('Test Time (seconds)')
    plt.legend()
    plt.grid(True)

    # test plot filename
    test_time_filename = f"{prefix}_{output_file_prefix}_average_test_time_vs_missing_ratio.png"
    test_time_path = os.path.join(output_images_dir, test_time_filename)
    print(f"Saving test time plot to: {test_time_path}")
    plt.savefig(test_time_path)
    plt.show()  

    print(f" Plots saved successfully: \n  - Train: {train_time_filename}\n  - Test: {test_time_filename}")
