#!/usr/bin/env python3
# File: src/load_data.py

# importing required libraries
import os
import pandas as pd
from src.utils import drop_all_nans
from src.config_loader import load_config
from src.utils import calculate_initial_missingness
from src.synthetic_data_generation import generate_synthetic_data
from src.dhs_modelling_functions_new import final_ds_droping_cols

def find_project_root():
    """
    Detect the project root where config.json exists.
    useful for fallback file loading when input_dir is not provided.
    """
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, 'config.json')):
        return current_dir
    else:
        # assuming this file is in src/, go up one level
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_file_dir)

def load_data(config):
    """
    Main data loading function that handles both real and synthetic datasets.
    - For synthetic: generates artificial data like original DHS numerical data based on correlation type selected by user.
    - For real data: loads pickled or CSV file based on configuration.
    applies standard preprocessing and optionally drops highly missing rows/surveys.
    """
    if config.get('use_synthetic_data', False):
        # for synthetic data: No 'process_nans' check needed
        print(f" Synthetic data selected with correlation type: {config['correlation_type']}")
        synthetic_data = generate_synthetic_data(
            config['correlation_type'], 
            config.get('dim', 96), 
            config.get('N', 15000)
        )
        return synthetic_data, calculate_initial_missingness(synthetic_data)

    # for real data: ensuring 'process_nans' is set
    if 'process_nans' not in config or not config['process_nans']:
        raise ValueError(" process_nans is missing in the configuration! Please ensure it's set before calling load_data().")
    print(f" process_nans in load_data(): {config['process_nans']}")

    # attempting to load real data from pickle file based on file names to look for in the expected folder
    print(" Loading real data...")
    dataset_type = config.get('dataset_type', 'HR')
    group_by_col = config.get('group_by_col', 'adm2_gaul')
    urban_rural_all_mode = config.get('urban_rural_all_mode', 'all')

    expected_pickle = f"5_grouped_df_V3_{dataset_type}_{group_by_col}_joined_with_ipc_{urban_rural_all_mode}.pkl"
    fallback_csv = "5_grouped_df_V3_HR_adm2_gaul_joined_with_ipc_all.csv"

    # trying to load from the given input directory if provided
    data_found = False
    if config.get('input_dir'):
        pickle_path = os.path.join(config['input_dir'], expected_pickle)
        if os.path.exists(pickle_path):
            print(f" Found dataset at: {pickle_path}")
            df = pd.read_pickle(pickle_path)
            data_found = True
        else:
            print(f"Dataset not found at: {pickle_path}")

    # if not found, fallback to a default location in /data/
    if not data_found:
        project_root = find_project_root()
        fallback_path = os.path.join(project_root, 'data', fallback_csv)
        if os.path.exists(fallback_path):
            print(f" Fallback: loading dataset from {fallback_path}")
            df = pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError(f"Dataset not found in fallback location: {fallback_path}")

    # applying dataset preprocessing: dropping meta columns, food help cols that are not necessary
    df = final_ds_droping_cols(df, 
                               drop_meta=True, drop_food_help=True, drop_perc=40,
                               retain_month=False, drop_highly_correlated_cols=False, drop_region=True, 
                               drop_data_sets=['Meta one-hot encoding', 'Meta frequency encoding'], 
                               use_NAN_amount_and_replace_NANs_in_categorical=False, 
                               drop_agricultural_cols=config['drop_agriculture'], 
                               drop_below_version=False, numerical_data=['mean'], retain_adm=False, 
                               retain_GEID_init=False, verbose=3)
    
    # again data cleaning by dropping specific FS and translation columns that are not needed
    drop_cols = [c for c in df.columns if 'FS;' in c and '0-2y' not in c]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.drop(columns=['DHS Cat; translator used: not at all', 'DHS Cat; translator used: yes'], inplace=True, errors='ignore')

    # dropping selected countries like "Egypt", "Comoros", "Central African Republic" as these lower overall data quality for its missingness
    if config['egypt_dropping']:
        df = df[~df['Meta; adm0_gaul'].isin(config['countries_to_drop'])]

    # if user selects 'drop_all_nans' from prompt, it will handle NaN values by dropping all NaNs 
    if config['process_nans'] == 'drop_all_nans':
        df = drop_all_nans(df)

    elif config['process_nans'] == 'numerical_only':
        # if 'numerical_only', just drop Egypt and return without further processing
        return df, initial_missingness
        
    # if user selects 'numerical_only_drop_20_percentage_nans' from prompt, it will handle NaN values by dropping 20% of NaNs only 
    elif config['process_nans'] == 'numerical_only_drop_20_percentage_nans':
        if 'Meta; GEID_init' in df.columns:
            survey_missingness = df.groupby('Meta; GEID_init').apply(lambda x: x.isna().mean().mean())
            surveys_to_keep = survey_missingness[survey_missingness <= 0.2].index
            df = df[df['Meta; GEID_init'].isin(surveys_to_keep)]
            
    # returning the final cleaned dataset and its initial missingness stats
    return df, calculate_initial_missingness(df)

