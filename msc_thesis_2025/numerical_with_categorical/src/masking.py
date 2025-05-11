#!/usr/bin/env python3
# File: src/masking.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import numpy as np
import pandas as pd

def apply_masking_for_cat_num(X_data, masking, missingness_fraction=0.3):
    """
    Apply MAR or MCAR missingness to a given DataFrame.

    Parameters:
    - X_data: pd.DataFrame, original dataset.
    - masking: bool, True = MAR, False = MCAR.
    - missingness_fraction: float, fraction of values to mask.

    Returns:
    - Modified DataFrame with NaNs introduced.
    - missing_mask: Boolean array marking masked positions.
    """
    import numpy as np

    X_data = X_data.copy()  # Prevent modifying original data
    missing_mask = np.zeros(X_data.shape, dtype=bool)  # Mask to track missing entries

    # Columns that MAR logic will condition on (only if present)
    condition_columns = [
        'type of place of residence',
        'has electricity',
        'has bank account',
        'owns land usable for agriculture'
    ]
    condition_columns = [col for col in condition_columns if col in X_data.columns]

    if masking:
        print("🔶 Applying MAR masking based on fixed socio-economic conditions...")

        # Apply missingness only to columns not used as MAR conditions
        target_columns = [col for col in X_data.columns if col not in condition_columns]

        # Define MAR condition: mask is applied where any of these conditions hold
        conditions = (
            (X_data.get('type of place of residence') == 'rural') |
            (X_data.get('has electricity') == 0) |
            (X_data.get('has bank account') == 0) |
            (X_data.get('owns land usable for agriculture') == 1)
        )

        eligible_indices = X_data[conditions].index  # Eligible rows for MAR masking

        # Loop over each target column to apply MAR-style missingness
        for col in target_columns:
            if col not in X_data.columns:
                continue

            non_missing = X_data[col].dropna().index
            col_eligible = non_missing.intersection(eligible_indices)
            num_to_mask = int(np.ceil(missingness_fraction * len(col_eligible)))

            if num_to_mask == 0:
                continue

            # Randomly select rows to mask
            sample_indices = np.random.choice(col_eligible, size=num_to_mask, replace=False)
            local_idx = X_data.index.get_indexer(sample_indices)

            # Update missing mask and set values to NaN
            missing_mask[local_idx, X_data.columns.get_loc(col)] = True
            X_data.loc[sample_indices, col] = np.nan

    else:
        print("Applying MCAR masking (completely random)...")

        # Apply random masking independently to each column
        for col in X_data.columns:
            non_missing_indices = X_data[col].dropna().index
            num_to_mask = int(np.ceil(missingness_fraction * len(non_missing_indices)))

            if num_to_mask == 0:
                continue

            random_sample = np.random.choice(non_missing_indices, size=num_to_mask, replace=False)
            local_indices = X_data.index.get_indexer(random_sample)

            # Apply random NaN and update missing mask
            missing_mask[local_indices, X_data.columns.get_loc(col)] = True
            X_data.loc[random_sample, col] = np.nan

    return X_data, missing_mask
    
# def apply_masking_for_cat_num(X_data, masking, missingness_fraction):
#     """
#     Apply random missingness to the dataset, with MAR logic applied 
#     when 'type of place of residence' is 'rural'.
    
#     Parameters:
#     - X_data: DataFrame containing the data.
#     - masking: Boolean indicating whether to apply masking.
#     - missingness_fraction: Fraction of data to be set as NaN.
    
#     Returns:
#     - X_data: DataFrame with applied missingness.
#     - missing_mask: Boolean array indicating where values are missing.
#     """
    
#     missing_mask = np.zeros(X_data.shape, dtype=bool)
    
#     if masking:
#         # Get the indices where 'type of place of residence' is 'rural'
#         rural_indices = X_data[X_data['type of place of residence'] == 'rural'].index

#         # Iterate through each column (except 'type of place of residence')
#         for col in X_data.columns:
#             if col == 'type of place of residence':
#                 continue  # Skip the target column itself
            
#             non_missing_indices = X_data[col].dropna().index
            
#             # Calculate the intersection of non-missing and rural indices
#             eligible_indices = non_missing_indices.intersection(rural_indices)
            
#             # Calculate number of samples to mask
#             num_to_mask = int(np.ceil(missingness_fraction * len(eligible_indices)))
#             if num_to_mask == 0:
#                 continue
            
#             # Randomly select from eligible indices (i.e., rows where 'rural' and not already NaN)
#             random_sample = np.random.choice(eligible_indices, size=num_to_mask, replace=False)
#             local_indices = X_data.index.get_indexer(random_sample)
            
#             # Mask the selected values
#             missing_mask[local_indices, X_data.columns.get_loc(col)] = True
#             X_data.loc[random_sample, col] = np.nan

#     else:
#         # Apply random masking (MCAR) to all columns
#         for col in X_data.columns:
#             non_missing_indices = X_data[col].dropna().index
#             random_sample = np.random.choice(non_missing_indices, size=int(np.ceil(missingness_fraction * len(non_missing_indices))), replace=False)
#             local_indices = X_data.index.get_indexer(random_sample)
#             missing_mask[local_indices, X_data.columns.get_loc(col)] = True
#             X_data.loc[random_sample, col] = np.nan
            
#     return X_data, missing_mask

def extract_values_using_mask(data, mask):
    """
    Extract values from the data where the mask is True.
    
    Parameters:
    - data (pd.DataFrame or np.ndarray): Input data from which values are extracted.
    - mask (same shape as data): Boolean mask indicating where to extract values.
    
    Returns:
    - pd.Series or np.ndarray: Flattened set of extracted non-null values.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape != mask.shape:
            raise ValueError(f"Data shape: {data.shape}, Mask shape: {mask.shape}")
            
        # extracting and returning values where the mask is True (excluding NaNs)
        extracted_values = data[mask]
        return extracted_values.dropna()  # ensuring only non-NaN rows are kept

    elif isinstance(data, np.ndarray):
        if data.shape != mask.shape:
            raise ValueError("The shape of data and mask do not match.")
        # direct masked extraction for numpy arrays
        return data[mask]

    else:
        raise TypeError("Data must be either a pandas DataFrame or numpy array.")
