#!/usr/bin/env python3
# File: src/load_data.py

# importing required libraries
import os
import pandas as pd
from pathlib import Path
from src.utils import drop_all_nans
from src.config_loader import load_config
from src.utils import calculate_initial_missingness
from src.synthetic_data_generation import generate_synthetic_data
from src.dhs_modelling_functions_new import final_ds_droping_cols

def find_project_root():
    """
    moving up from this file's directory until it finds 'config.json'.
    Raises FileNotFoundError if it hits the filesystem root without finding it.
    """
    current = Path(__file__).resolve().parent

    # climbing until finding config.json or reaching the filesystem root
    while True:
        if (current / 'config.json').exists():
            return current
        if current.parent == current:
            # if reached the top and never saw config.json
            raise FileNotFoundError(f"Could not find '{root_marker}' in any parent of {__file__}")
        current = current.parent

def load_data(config):
    """
    Main data loading function that handles both real and synthetic datasets.
    - For synthetic: generates artificial data like original DHS numerical data based on correlation type selected by user.
    - For real data: loads pickled or CSV file based on configuration.
    Applies standard preprocessing and optionally drops highly missing rows/surveys.
    """

    # input_dir to an absolute Path
    input_dir = Path(config.get('input_dir', 'data')).expanduser().resolve()
    config['input_dir'] = str(input_dir)
    
    # case 1: Synthetic data
    if config.get('use_synthetic_data', False):
        print(f" Synthetic data selected with correlation type: {config['correlation_type']}")
        synthetic_data = generate_synthetic_data(
            config['correlation_type'], 
            config.get('dim', 96), 
            config.get('N', 15000)
        )
        return synthetic_data, calculate_initial_missingness(synthetic_data)

    # case 2: Real data - validate 'process_nans' from prompt
    # for real data: ensuring 'process_nans' is set
    if 'process_nans' not in config or not isinstance(config['process_nans'], str):
        raise ValueError("'process_nans' is missing or invalid in the config. Please ensure it's selected by the user.")

    selected_option = config['process_nans']
    print(f" process_nans selected by user: {selected_option}")

    # to load real data from pickle file based on file names to look for in the expected folder
    print("Loading real DHS dataset...")
    dataset_type = config.get('dataset_type', 'HR')
    group_by_col = config.get('group_by_col', 'adm2_gaul')
    urban_rural_all_mode = config.get('urban_rural_all_mode', 'all')

    expected_pickle = f"5_grouped_df_V3_{dataset_type}_{group_by_col}_joined_with_ipc_{urban_rural_all_mode}.pkl"
    fallback_csv = "5_grouped_df_V3_HR_adm2_gaul_joined_with_ipc_all.csv"

    # trying to load from the given input directory if provided
    df = None
    pk = input_dir / expected_pickle
    if pk.exists():
        print(f"Found dataset at: {pk}")
        df = pd.read_pickle(pk)

    # fallback to <project_root>/data/
    if df is None:
        project_root = find_project_root()
        fb = project_root / 'data' / fallback_csv
        if fb.exists():
            print(f"Fallback: loading dataset from {fb}")
            df = pd.read_csv(fb)
        else:
            raise FileNotFoundError(f"No dataset found at either {pk} or {fb}")

    # preprocessing
    # applying dataset preprocessing: dropping meta columns, food help cols that are not necessary
    df = final_ds_droping_cols(
        df,
        drop_meta=True, drop_food_help=True, drop_perc=40,
        retain_month=False, drop_highly_correlated_cols=False, drop_region=True,
        drop_data_sets=['Meta one-hot encoding', 'Meta frequency encoding'],
        use_NAN_amount_and_replace_NANs_in_categorical=False,
        drop_agricultural_cols=config.get('drop_agriculture', False),
        drop_below_version=False, numerical_data=['mean'], retain_adm=False,
        retain_GEID_init=False, verbose=3
    )

    # again data cleaning by dropping specific FS and translation columns that are not needed
    drop_cols = [c for c in df.columns if 'FS;' in c and '0-2y' not in c]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.drop(columns=['DHS Cat; translator used: not at all', 'DHS Cat; translator used: yes'], inplace=True, errors='ignore')

    # dropping selected countries like "Egypt", "Comoros", "Central African Republic" as these reduce the overall data quality for its missingness
    if config.get('egypt_dropping', False):
        df = df[~df['Meta; adm0_gaul'].isin(config.get('countries_to_drop', []))]

    # handling missingness based on user selection
    # if user selects 'drop_all_nans' from prompt, it will handle NaN values by dropping all NaNs in the dataset
    if selected_option == 'drop_all_nans':
        df = drop_all_nans(df)
        
    # elif config['process_nans'] == 'numerical_only':
    elif selected_option == 'keep_all_numerical':
        # if 'keep_all_numerical', just dropping Egypt and returning without further processing
        return df, calculate_initial_missingness(df)

    # if user selects 'numerical_only_drop_20_percentage_nans' from prompt, it will handle NaN values by dropping 20% of NaNs only 
    elif selected_option == 'numerical_only_drop_20_percentage_nans':
        if 'Meta; GEID_init' in df.columns:
            survey_missingness = df.groupby('Meta; GEID_init').apply(lambda x: x.isna().mean().mean())
            surveys_to_keep = survey_missingness[survey_missingness <= 0.2].index
            df = df[df['Meta; GEID_init'].isin(surveys_to_keep)]

    # returning cleaned dataset and initial missingness stats
    return df, calculate_initial_missingness(df)
