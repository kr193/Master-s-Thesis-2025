#!/usr/bin/env python3
# File: /root/main.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import pandas as pd
from src.utils import *
from src.evaluation import *
from src.run_pipeline import *
from src.visualization import *
from src.gain import GAIN_code_v2
from src.load_data import load_data
from src.deep_learning_methods import *
from src.config_loader import load_config
from src.setup_dirs import setup_directories
from src.config_loader import load_config, get_final_imputations_dir
from src.dhs_modelling_functions_new import final_ds_droping_cols, fold_generator
from src.helper_functions import select_process_nans_option, select_masking_option

# important columns
important_columns = [
    'DHS Cat; source of drinking water: piped into dwelling',
    'DHS Num; cluster altitude in meters: mean',
    'DHS Num; number of mosquito bed nets: mean',
    'DHS Num; time to get to water source (minutes): mean',
    'DHS Cat; location of source for water: in own dwelling',
    'DHS Cat; type of toilet facility: flush to piped sewer system',
    'DHS Num; number of household members: mean',
    'DHS Cat; has mobile telephone: yes',
    'DHS Cat; has television: yes',
    'DHS Cat; type of cooking fuel: lpg',
    'DHS Num; hectares of agricultural land (1 decimal): mean',
    'DHS Num; owns sheep: mean',
    'DHS Num; total adults measured: mean'
]

def main():
    # loading config from json file, edit if necessary
    print("\n Starting Evaluation (Task 1 and Task3)...")
    config = load_config()
    # a prompt is displayed to the user, waiting for user input for the 'process_nans' selection
    config = select_process_nans_option(config)
    print(f" The selected process_nans: {config['process_nans']}")
    # again, prompt user for masking selection (either select MAR or MCAR)
    config['masking'] = select_masking_option()
    config['output_file_prefix'] = 'with_masking' if config['masking'] else 'without_masking'
    print(f" The selected missingness type: {config['output_file_prefix']}")
    # setup directories (which needs process_nans and masking)
    config = setup_directories(config)
    # loading dataset from /data folder
    input_df, initial_missingness = load_data(config)
    
    # defining the different imputers and missing ratios 
    imputers = ['Mean', 'KNN', 'MICE_Ridge', 'AE', 'DAE', 'VAE', 'GAIN']
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    # loading or computing data and saving pickles for Task 1 and Task 3
    load_or_compute_data_part(input_df, imputers, missing_ratios, config, config['masking'])
    # evaluating metrics using saved pickles
    final_results, extracted_values, time_metrics = evaluate_metrics_part(input_df, imputers, missing_ratios, config)
    # plotting RMSE statistics
    missing_ratio_vs_stats(final_results, imputers, missing_ratios, config)
    # saving time metrics to Excel
    save_time_metrics_to_excel(time_metrics, config)
    # plotting time metrics (for visualizing training and test time per imputation method)
    plot_time_vs_missing_ratio(time_metrics, config)
    # scatter plots for the entire DataFrame
    combined_df_scatter_plots(extracted_values, config, missing_ratio=0.3)
    # per column scatter plots for the entire DataFrame
    per_column_scatter_plots(extracted_values, important_columns, imputers, config, missing_ratio=0.3)

    print("\n Starting Feature Evaluation (Task 2)...")
    # feature evaluation (RMSE vs Number of Features)
    config['output_file_prefix'] = 'with_masking' if config['masking'] else 'without_masking'
    # defining feature intervals for Task 2
    feature_intervals = [15, 30, 45, 60, 75, 96]
    # loading or computing data and saving pickles for Task 2
    load_or_compute_data_feature_evaluation(input_df, imputers, feature_intervals, config, config['masking'])
    # evaluate metrics for Task 2
    final_results_features = evaluate_metrics_part_feature_evaluation(input_df, imputers, feature_intervals, config)
    # plotting RMSE vs Number of Features
    num_of_features_vs_stats(final_results_features, imputers, feature_intervals, config)


if __name__ == "__main__":
    main()