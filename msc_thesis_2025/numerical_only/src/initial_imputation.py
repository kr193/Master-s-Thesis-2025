#!/usr/bin/env python3
# File: src/initial_imputation.py

# importing required libraries
import os
import pickle
import pandas as pd
from sklearn.impute import KNNImputer
from src.config_loader import get_final_imputations_dir

# --------------------------------------------------------------------------------------------------------------------------------------------
# Perform initial KNN imputation on raw DHS's numerical data (real/synthetic) containing missing values, which simulates real-world scenarios.
# Dataset with NaNs can not be scaled or normalized, so before scaling pre KNN imputation is executed on raw numerical dataset.
# This function handles pre-KNN imputation for train/val/test data by saving results and to avoid recomputation in future runs.
# --------------------------------------------------------------------------------------------------------------------------------------------

def first_knn_imp_for_numerical_data_with_missing(X_train_final, X_val_final, X_test_final, fold, masking_dir, missing_ratio, repeat, config):
    """
    Perform or load KNN 'initial' imputation for 'numerical data with missing' as 'Scaling' can not handle NaNs values.
    """
    _ , imputation_dir, _ , _ = get_final_imputations_dir(config)

    # file names include missing_ratio and repetition to differentiate between different scenarios
    train_imputed_file = os.path.join(imputation_dir, f"train_first_knn_imp_for_numerical_data_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    val_imputed_file = os.path.join(imputation_dir, f"val_first_knn_imp_for_numerical_data_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    test_imputed_file = os.path.join(imputation_dir, f"test_first_knn_imp_for_numerical_data_fold_{fold}_{missing_ratio}_repeat_{repeat}.pkl")

    if os.path.exists(train_imputed_file) and os.path.exists(val_imputed_file) and os.path.exists(test_imputed_file):
        print(f"Loading pre-imputed data for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")
        X_train_final = pickle.load(open(train_imputed_file, 'rb'))
        X_val_final = pickle.load(open(val_imputed_file, 'rb'))
        X_test_final = pickle.load(open(test_imputed_file, 'rb'))
    else:
        print(f"Applying KNN imputation for numerical data with initial nans for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")
        imputer = KNNImputer(n_neighbors=5)
        X_train_final = pd.DataFrame(imputer.fit_transform(X_train_final), index=X_train_final.index, columns=X_train_final.columns)
        X_val_final = pd.DataFrame(imputer.transform(X_val_final), index=X_val_final.index, columns=X_val_final.columns)
        X_test_final = pd.DataFrame(imputer.transform(X_test_final), index=X_test_final.index, columns=X_test_final.columns)

        pickle.dump(X_train_final, open(train_imputed_file, 'wb'))
        pickle.dump(X_val_final, open(val_imputed_file, 'wb'))
        pickle.dump(X_test_final, open(test_imputed_file, 'wb'))
    
    return X_train_final, X_val_final, X_test_final

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Prepares KNN-Imputed Validation/Test Sets for AE methods only as Validation/Test Sets are having artificial missingness to stimulate a real-world scenario.
# Trains or loads a KNN imputer and applies it to validation/test data before passing to AE, DAE or VAE methods.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def initial_knn_imputed_data_for_aes(X_train_scaled, X_val_with_missing, X_test_with_missing, fold, masking_dir, missing_ratio, repeat, config):
    """
    Function to train or load KNN Imputer for each fold and repetition, and save/load the imputed data, considering missing ratios.

    Train or load KNN Imputer and save/load the imputed validation and test data.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training data.
        X_val_with_missing (pd.DataFrame): Validation data with missing values.
        X_test_with_missing (pd.DataFrame): Test data with missing values.
        fold (int): Fold number for cross-validation.
        masking_dir (str): Directory to save or load imputed data.
        missing_ratio (float): Ratio of missing data in the dataset.
        repeat (int): Repetition number for the experiment.

    Returns:
        tuple: Imputed validation and test data as DataFrames.
    """
    _, imputation_dir_knn, _ , _ = get_final_imputations_dir(config)
   
    imputer_name = f'KNN_Imputer_for_aes_Fold_{missing_ratio}_fold_{fold}_repeat_{repeat}'
    imputation_file = os.path.join(imputation_dir_knn, f"{imputer_name}.pkl")

    # paths for saving imputed validation and test data
    val_imputed_file = os.path.join(imputation_dir_knn, f"X_val_knn_imputed_aes_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    test_imputed_file = os.path.join(imputation_dir_knn, f"X_test_knn_imputed_aes_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")

    # if the imputed data already exists, load it directly
    if os.path.exists(val_imputed_file) and os.path.exists(test_imputed_file):
        print(f"Loading initial KNN imputed data for aes for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")

        # loading the imputed validation data
        with open(val_imputed_file, 'rb') as f:
            X_val_imputed = pickle.load(f)

        # loading the imputed test data
        with open(test_imputed_file, 'rb') as f:
            X_test_imputed = pickle.load(f)
            
        print(f"Imputed data loaded for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}.")
        return X_val_imputed, X_test_imputed

    # if the imputer exists, loading it
    if os.path.exists(imputation_file):
        with open(imputation_file, 'rb') as f:
            knn_imputer_fold = pickle.load(f)
        print(f"Loaded initial KNN imputer for aes for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}.")
    else:
        # training the KNN imputer if not already saved
        print(f"Training initial KNN imputer for aes  for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")
        knn_imputer_fold = KNNImputer(n_neighbors=5)
        knn_imputer_fold.fit(X_train_scaled)
        
        # saving the trained KNN imputer
        with open(imputation_file, 'wb') as f:
            pickle.dump(knn_imputer_fold, f)
        print(f"initial KNN imputer for aes trained and saved for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}.")

    # imputing validation and test data
    print(f"Imputing validation and test data for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")
    X_val_imputed = pd.DataFrame(knn_imputer_fold.transform(X_val_with_missing.values), 
                                 index=X_val_with_missing.index, columns=X_val_with_missing.columns)
    X_test_imputed = pd.DataFrame(knn_imputer_fold.transform(X_test_with_missing.values), 
                                  index=X_test_with_missing.index, columns=X_test_with_missing.columns)

    print(f"initial KNN Imputed data for aes saved for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}.")
    
    return X_val_imputed, X_test_imputed

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Apply initial KNN imputation on raw DHS's numerical data (real/synthetic) containing missing values, which simulates real-world scenarios for Feature Evaluation Task (Task 2).
# used when analysing how performance changes with increasing features
# handles train/val/test splits and caches the outputs.
# Dataset with NaNs can not be scaled or normalised, so before scaling, pre-KNN imputation is executed on the raw numerical dataset.
# This function handles pre-KNN imputation for train/val/test data by saving results and to avoid recomputation in future runs.
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def first_knn_imp_for_numerical_data_with_missing_feature_evaluation(X_train_final, X_val_final, X_test_final, fold, final_methods_dir, n_features, repeat, config):
    """
    Function to perform or load KNN imputation for numerical data, considering the number of features and repetition.
    """

    _ , feature_eval_dir, _ , _ = get_final_imputations_dir(config)

    # file names include n_features and repetition to differentiate between different scenarios
    train_imputed_file = os.path.join(feature_eval_dir, f"train_knn_imp_num_features_{n_features}_fold_{fold}_repeat_{repeat}.pkl")
    val_imputed_file = os.path.join(feature_eval_dir, f"val_knn_imp_num_features_{n_features}_fold_{fold}_repeat_{repeat}.pkl")
    test_imputed_file = os.path.join(feature_eval_dir, f"test_knn_imp_num_features_{n_features}_fold_{fold}_repeat_{repeat}.pkl")

    if os.path.exists(train_imputed_file) and os.path.exists(val_imputed_file) and os.path.exists(test_imputed_file):
        print(f"Loading pre-imputed data for fold {fold}, number of features {n_features}, repetition {repeat}...")
        X_train_final = pickle.load(open(train_imputed_file, 'rb'))
        X_val_final = pickle.load(open(val_imputed_file, 'rb'))
        X_test_final = pickle.load(open(test_imputed_file, 'rb'))
    else:
        print(f"Applying KNN imputation for numerical data for fold {fold}, number of features {n_features}, repetition {repeat}...")
        imputer = KNNImputer(n_neighbors=5)
        X_train_final = pd.DataFrame(imputer.fit_transform(X_train_final), index=X_train_final.index, columns=X_train_final.columns)
        X_val_final = pd.DataFrame(imputer.transform(X_val_final), index=X_val_final.index, columns=X_val_final.columns)
        X_test_final = pd.DataFrame(imputer.transform(X_test_final), index=X_test_final.index, columns=X_test_final.columns)

        pickle.dump(X_train_final, open(train_imputed_file, 'wb'))
        pickle.dump(X_val_final, open(val_imputed_file, 'wb'))
        pickle.dump(X_test_final, open(test_imputed_file, 'wb'))

    return X_train_final, X_val_final, X_test_final

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Generates KNN-Imputed Validation/Test Sets for AE methods only as Validation/Test Sets are having artificial missingness to stimulate a real-world scenario in Feature Evaluation.
# Trains or loads a KNN imputer and applies it to validation/test data before passing to AE, DAE or VAE methods.
# Saves/Reuses methods and imputed values per fold and repeat.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def initial_knn_imputed_data_for_aes_feature_evaluation(X_train_scaled, X_val_with_missing, X_test_with_missing, fold, final_methods_dir, n_features, repeat, config):
    """
    Function to train or load KNN Imputer for each fold and repetition, and save/load the imputed data, considering the number of features.
    """
    _ , feature_eval_dir, _ , _ = get_final_imputations_dir(config)

    imputer_name = f'KNN_Imputer_{n_features}_features_fold_{fold}_repeat_{repeat}'
    imputation_file = os.path.join(feature_eval_dir, f"{imputer_name}.pkl")

    val_imputed_file = os.path.join(feature_eval_dir, f"X_val_knn_imputed_features_{n_features}_fold_{fold}_repeat_{repeat}.pkl")
    test_imputed_file = os.path.join(feature_eval_dir, f"X_test_knn_imputed_features_{n_features}_fold_{fold}_repeat_{repeat}.pkl")

    if os.path.exists(val_imputed_file) and os.path.exists(test_imputed_file):
        print(f"Loading KNN imputed data for fold {fold}, number of features {n_features}, repetition {repeat}...")
        X_val_imputed = pickle.load(open(val_imputed_file, 'rb'))
        X_test_imputed = pickle.load(open(test_imputed_file, 'rb'))
        return X_val_imputed, X_test_imputed

    if os.path.exists(imputation_file):
        knn_imputer_fold = pickle.load(open(imputation_file, 'rb'))
        print(f"Loaded KNN imputer for fold {fold}, number of features {n_features}, repetition {repeat}.")
    else:
        print(f"Training KNN imputer for fold {fold}, number of features {n_features}, repetition {repeat}...")
        knn_imputer_fold = KNNImputer(n_neighbors=5)
        knn_imputer_fold.fit(X_train_scaled)
        pickle.dump(knn_imputer_fold, open(imputation_file, 'wb'))

    X_val_imputed = pd.DataFrame(knn_imputer_fold.transform(X_val_with_missing.values), 
                                 index=X_val_with_missing.index, columns=X_val_with_missing.columns)
    X_test_imputed = pd.DataFrame(knn_imputer_fold.transform(X_test_with_missing.values), 
                                  index=X_test_with_missing.index, columns=X_test_with_missing.columns)

    pickle.dump(X_val_imputed, open(val_imputed_file, 'wb'))
    pickle.dump(X_test_imputed, open(test_imputed_file, 'wb'))

    return X_val_imputed, X_test_imputed