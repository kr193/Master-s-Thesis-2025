#!/usr/bin/env python3
# File: src/utils.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K

# loading from disk 
def get_final_imputations_dir(config):
    """
    Dynamically constructs and returns the final_imputations_dir_missing_ratio based on config.
    Ensures directory is created and avoids hardcoding or global variables.
    """
    masking_dir = os.path.join(config['base_dir'], 'with_masking' if config['masking'] else 'without_masking')
    final_methods_dir = os.path.join(masking_dir, 'final_methods_pkls')
    final_imputations_dir = os.path.join(final_methods_dir, 'task1_final_imputations_missing_ratio')
    os.makedirs(final_imputations_dir, exist_ok=True)
    return final_imputations_dir

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# to calculate initial missingness
def calculate_initial_missingness(df):
    total_elements = df.size
    missing_elements = df.isnull().sum().sum()
    return missing_elements / total_elements

def save_pickle(file_path, data):
    """
    Save data to a pickle file with the highest protocol for efficiency.

    Parameters:
    - file_path (str): Destination path to save the pickle file.
    - data: Python object to be serialized and saved.

    Returns:
    - None
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
    
# global cache to reduce disk reads during repeated evaluations
imputed_data_cache = {}

def load_imputed_data(imputer_name, missing_ratio, fold, config):
    """
    Loads imputed data from cache if available, otherwise from disk.
    
    Parameters:
    - imputer_name: Name of the imputation method.
    - missing_ratio: Missingness ratio.
    - fold: Fold index.

    Returns:
    - The imputed dataset or None if not found.
    """
    # using a tuple key to uniquely identify this imputer-fold-ratio combo
    cache_key = (imputer_name, missing_ratio, fold)
    if cache_key in imputed_data_cache:
        return imputed_data_cache[cache_key]  # returning cached result

    
    final_imputations_dir_missing_ratio = get_final_imputations_dir(config)
    filename = os.path.join(final_imputations_dir_missing_ratio, f"{imputer_name}_imputed_{missing_ratio}_fold_{fold}_repeat_0.pkl")

    print(f"Trying to load imputed data from: {filename}")
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            imputed_data_cache[cache_key] = data  # storing in cache
            return data

    print(f"File does not exist: {filename}")
    return None

def safe_load_pickle(file_path):
    """
    safely loads pickle file and returning None if an error occurs.
    
    Parameters:
    - file_path: Path to the pickle file.
    
    Returns:
    - Loaded data or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def save_data(file_path, data):
    """
    Saves the data (train, val or test) to a file using pickle.
    """ 
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    """
    Loads the data from a pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def convert_sparse_to_dense(df):
    """
    Convert sparse columns in a pandas DataFrame to dense format.

    Parameters:
    df : pandas.DataFrame
        Input DataFrame potentially containing sparse columns.

    Returns:
    pandas.DataFrame
        DataFrame with all sparse columns converted to dense format.
    """
    # applying conversion: check each column's dtype, converting sparse columns to dense
    return df.apply(lambda col: col.sparse.to_dense() if isinstance(col.dtype, pd.SparseDtype) else col)

def decode_one_hot_cached(imputed_df, encoder, categorical_columns, fold, imputer_name, missing_ratio, cache_dir, binary_threshold=2, binarize_threshold=0.6):
    """
    Decode one-hot encoded data with caching to avoid redundant decoding.

    Parameters:
    - imputed_df: DataFrame with one-hot encoded predictions
    - encoder: fitted OneHotEncoder
    - categorical_columns: original categorical column names
    - fold: fold number
    - imputer_name: name of the imputer
    - missing_ratio: missing ratio
    - cache_dir: directory to store cached decoded outputs
    - binary_threshold: number of categories to be treated as binary
    - binarize_threshold: threshold for converting soft values (e.g., 0.6 → 1, else 0)

    Returns:
    - df_decoded: Decoded DataFrame with original categorical variables restored.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"decoded_{imputer_name}_{missing_ratio}_fold_{fold}.pkl")
    if os.path.exists(cache_file):
        print(f"Loaded cached decoded data: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    df_encoded = imputed_df.copy()
    df_decoded = df_encoded.copy()
    categorical_columns = list(categorical_columns)

    # mapping each categorical column to its corresponding one-hot encoded columns
    encoded_col_names = encoder.get_feature_names_out(categorical_columns)

    category_mapping = {}
    for i, col in enumerate(categorical_columns):
        start = sum(len(cats) for cats in encoder.categories_[:i])
        end = start + len(encoder.categories_[i])
        category_mapping[col] = encoded_col_names[start:end]

    # decoding each group of one-hot columns back into a single categorical column
    for col, encoded_cols in category_mapping.items():
        # binary class prediction: Apply hard threshold
        sub_df = df_encoded[encoded_cols].copy()

        if len(encoded_cols) <= binary_threshold:
            sub_df = (sub_df > binarize_threshold).astype(int)

        # argmax selection (max probability = predicted class)
        decoded_series = sub_df.idxmax(axis=1).apply(lambda x: x.split('_')[-1])

        categories = encoder.categories_[categorical_columns.index(col)]

        # ensuring proper type casting (int for binary features)
        if all(cat in ['0', '1'] or isinstance(cat, (int, float)) for cat in categories):
            try:
                decoded_series = decoded_series.astype(int)
            except ValueError:
                decoded_series = decoded_series.astype(str)
        else:
            decoded_series = decoded_series.astype(str)

        df_decoded[col] = decoded_series
        
    # dropping the one-hot encoded columns after decoding
    df_decoded.drop(columns=encoded_col_names, inplace=True)

    # saving decoded result to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(df_decoded, f)
        print(f"💾 Cached decoded data to: {cache_file}")

    return df_decoded

def adapt_original_mask_to_one_hot(original_data, original_mask):
    """
    Adapt a categorical mask to align with one-hot encoded columns, 
    preserving the original missing data pattern in the encoded data structure.

    Parameters:
    - original_data: DataFrame with one-hot encoded columns.
    - original_mask: Original mask DataFrame with categorical columns.

    Returns:
    - one_hot_mask: Boolean DataFrame mask matching the one-hot encoded structure.
    """
    # starting with all False
    one_hot_mask = pd.DataFrame(False, index=original_data.index, columns=original_data.columns)

    # efficient bulk assignment using a dictionary to avoid fragmentation
    new_columns_data = {}

    for col in original_mask.columns:
        if col in original_data.columns:
            new_columns_data[col] = original_mask[col]
        else:
            # identifying corresponding one-hot columns
            one_hot_cols = [c for c in original_data.columns if c.startswith(col + "_")]
            for one_hot_col in one_hot_cols:
                new_columns_data[one_hot_col] = original_mask[col]

    # using pd.concat for performance and avoid fragmentation
    updates_df = pd.DataFrame(new_columns_data, index=original_data.index)
    one_hot_mask.update(updates_df)

    # fixing FutureWarning from .fillna and .astype on object dtype
    one_hot_mask = one_hot_mask.fillna(False)
    one_hot_mask = one_hot_mask.infer_objects(copy=False).astype(bool)

    return one_hot_mask

def add_noise_to_categorical(X_data, categorical_columns, noise_level=0.1):
    """
    Add random noise to categorical columns by swapping values within each categorical column.
    This simulates data corruption or noise for robustness testing.

    Parameters:
    - X_data: Input DataFrame containing categorical data.
    - categorical_columns (list): List of categorical column names to add noise to.
    - noise_level (float): Fraction of entries to corrupt (default: 0.1, i.e., 10%).

    Returns:
    - X_data_noisy: DataFrame with noise added to categorical columns.
    """
    # a copy of the original data to preserve it
    X_data_noisy = X_data.copy()
    
    # iterating over each categorical column to introduce noise
    for col in categorical_columns:
        # determining the number of entries to alter based on the specified noise level
        n_values_to_modify = int(noise_level * len(X_data_noisy))
        
        # randomly select indices that will be affected by noise
        indices_to_modify = np.random.choice(X_data_noisy.index, size=n_values_to_modify, replace=False)
        
        # shuffle selected indices to swap values randomly
        shuffled_indices = np.random.permutation(indices_to_modify)

        # swapping values at selected indices within the column
        X_data_noisy.loc[indices_to_modify, col] = X_data_noisy.loc[shuffled_indices, col].values
    
    return X_data_noisy

def separate_categorical_mask(original_data, original_mask):
    """
    Extract a mask specifically for categorical columns from the provided original mask.
    Ensures alignment between original data and the categorical mask.

    Parameters:
    - original_data: Original DataFrame containing both numerical and categorical columns.
    - original_mask (pd.DataFrame or np.ndarray): Boolean mask indicating missing positions.

    Returns:
    - categorical_mask (pd.DataFrame): Mask indicating missingness specifically for categorical columns.
    """
    
    # identifying categorical columns based on data types
    categorical_columns = original_data.select_dtypes(include=['object', 'category']).columns

    # ensuring that `original_mask` is a DataFrame with matching structure
    if isinstance(original_mask, np.ndarray):
        # converting to DataFrame using the original data's structure
        original_mask = pd.DataFrame(original_mask, columns=original_data.columns)

    # extracting the part of the mask that corresponds to categorical columns
    categorical_mask = original_mask[categorical_columns]

    return categorical_mask


