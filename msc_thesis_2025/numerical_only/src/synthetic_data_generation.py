#!/usr/bin/env python3
# File: src/synthetic_data_generation.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import numpy as np
import pandas as pd

# generates synthetic datasets like raw DHS (numerical+aggregated) dataset with varying correlation levels
def generate_synthetic_data(correlation_type, dim=96, N=15000):
    """
    Generate synthetic data with a combination of base columns and subcategories.
    
    This function creates synthetic data based on the specified correlation type, dimensionality,
    and number of samples. It generates columns with a mix of base categories and subcategories 
    and ensures that the output dataset has the desired correlation structure.

    Args:
        correlation_type (str): The type of correlation to introduce in the data. 
                                Options are:
                                - 'no_correlation': No correlation between columns.
                                - 'medium_correlation': Medium level of correlation between columns.
                                - 'high_correlation': High level of correlation between columns.
        dim (int, optional): The total number of dimensions (columns) for the synthetic data. 
                             Defaults to 96.
        N (int, optional): The number of rows (samples) to generate. Defaults to 15000.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the synthetic data with the specified
                      correlation type and structure.

    Raises:
        ValueError: If the `correlation_type` is not one of the allowed options.

    Example:
        >>> df = generate_synthetic_data(correlation_type='medium_correlation', dim=96, N=10000)
        >>> print(df.head())
    """
    np.random.seed(6688)

    # base columns without subcategories (total of 46)
    base_columns = [f"{chr(65+i)}{chr(97+i)}" for i in range(46)]  # Aa, Bb, Cc, ..., upto 46

    # base columns with subcategories (50 subcategories total)
    columns_with_subcategories = {
        'Ab': [f'a{i+1}' for i in range(15)],    # 15 subcategories
        'Bc': [f'b{i+1}' for i in range(13)],    # 13 subcategories
        'Cd': [f'c{i+1}' for i in range(9)],     # 9 subcategories
        'Dd': [f'd{i+1}' for i in range(3)],     # 3 subcategories
        'Ee': [f'e{i+1}' for i in range(3)],     # 3 subcategories
        'Ff': [f'f{i+1}' for i in range(3)],     # 3 subcategories
        'Gg': [f'g{i+1}' for i in range(4)]      # 4 subcategories
    }

    # building the column names
    synthetic_columns = []

    # adding the base columns (46 columns without subcategories)
    synthetic_columns.extend(base_columns)

    # adding columns with subcategories
    for base_col, subcategories in columns_with_subcategories.items():
        for sub in subcategories:
            synthetic_columns.append(f"{base_col}:{sub}")

    # ensuring the total number of columns is 96 (46 base columns + 50 subcategories)
    assert len(synthetic_columns) == 96, f"Total columns: {len(synthetic_columns)}"

    # creating the covariance matrix based on the correlation type
    if correlation_type == 'no_correlation':
        Sigma = np.diag(np.random.uniform(low=1.0, high=2.0, size=len(synthetic_columns)))
    elif correlation_type == 'medium_correlation':
        A = np.random.normal(size=(len(synthetic_columns), len(synthetic_columns)))
        Sigma = np.diag(np.ones(len(synthetic_columns))) * 0.5 + A @ A.T * 0.5
    elif correlation_type == 'high_correlation':
        A = np.random.normal(size=(len(synthetic_columns), len(synthetic_columns)))
        Sigma = np.diag(np.ones(len(synthetic_columns))) * 0.2 + A @ A.T * 0.8
    else:
        raise ValueError("Invalid correlation_type. Choose from 'no_correlation', 'medium_correlation', 'high_correlation'.")

    # generating multivariate normal data
    Mu = np.random.normal(loc=0.0, scale=2.0, size=len(synthetic_columns))
    data = np.random.multivariate_normal(mean=Mu, cov=Sigma, size=N)

    # returning the synthetic df with the defined columns
    return pd.DataFrame(data, columns=synthetic_columns)