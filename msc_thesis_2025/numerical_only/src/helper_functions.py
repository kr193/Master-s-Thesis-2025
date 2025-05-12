# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# ---------------------------------------------------------
# prompt: Select Data Processing Strategy
# used when real or synthetic data is involved
# ---------------------------------------------------------

def select_process_nans_option(config):
    """
    Prompts the user based on whether synthetic data is used.
    If synthetic: ask for correlation_type.
    If real data: ask for process_nans option.
    """
    if config.get('use_synthetic_data', False):
        # synthetic data: prompt for correlation type
        options = {
            '1': 'no_correlation',
            '2': 'medium_correlation',
            '3': 'high_correlation'
        }
        print("\nSelect correlation type for synthetic data:")
        print("1 - no_correlation")
        print("2 - medium_correlation")
        print("3 - high_correlation")
        
        selection = input("Enter option number (1/2/3): ").strip()
        while selection not in options:
            print("Invalid selection. Please enter 1, 2, or 3.")
            selection = input("Enter option number (1/2/3): ").strip()
        
        config['correlation_type'] = options[selection]
        print(f"You selected correlation_type: {config['correlation_type']}")

    else:
        # real data: prompt for process_nans
        options = {
            '1': 'keep_all_numerical',
            '2': 'drop_all_nans',
            '3': 'numerical_only_drop_20_percentage_nans'
        }
        print("\nSelect process_nans option:")
        print("1 - keep_all_numerical")
        print("2 - drop_all_nans")
        print("3 - drop_20_percentage (numerical_only_drop_20_percentage_nans)")
        
        selection = input("Enter option number (1/2/3): ").strip()
        while selection not in options:
            print("Invalid selection. Please enter 1, 2, or 3.")
            selection = input("Enter option number (1/2/3): ").strip()
        
        config['process_nans'] = options[selection]
        print(f" You selected process_nans: {config['process_nans']}")

    return config

# ---------------------------------------------------------
# Prompt: Select Missingness Type (MAR or MCAR)
# determines whether masking will be applied
# ---------------------------------------------------------

def select_masking_option():
    options = {
        '1': True,   # MAR (with masking)
        '2': False   # MCAR (without masking)
    }
    print("\nSelect missingness type:")
    print("1 - MAR (with masking)")
    print("2 - MCAR (without masking)")

    selection = input("Enter option number (1/2): ").strip()
    while selection not in options:
        print("Invalid selection. Please enter 1 or 2.")
        selection = input("Enter option number (1/2): ").strip()

    return options[selection]
