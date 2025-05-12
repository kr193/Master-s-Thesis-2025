#importing Libraries
import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from icecream import ic
import tensorflow as tf
from scipy.stats import skew
from tensorflow import keras
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras import Model
from collections import defaultdict
import tensorflow.keras.backend as K
from pandarallel import pandarallel
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from tensorflow.keras.losses import mse
from tensorflow.keras import Sequential
from sklearn.impute import SimpleImputer
from scipy.stats import levene, f_oneway
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.impute import IterativeImputer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import mse as keras_mse
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from tensorflow.keras.layers import Lambda, Layer, Input, Dense
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Dense, Input, GaussianNoise, Layer
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from tensorflow.keras import Sequential, layers, models, optimizers, regularizers
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures, FunctionTransformer

pandarallel.initialize()
from src.dhs_modelling_functions_new import final_ds_droping_cols, fold_generator
from src.layersconfig import *

def root_mean_squared_error(y_true, y_pred):
    '''
    This loss function is used for regression models during training.
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def apply_masking(X_data, masking, missingness_fraction):
    '''
    Applies either MAR-style masking (grouped by base column) or MCAR-style random masking.
    Returns the masked data and corresponding mask matrix.
    '''
    X_data=X_data.copy()
    # print("Original DataFrame Columns:", X_data.shape) 
    # print("Original DataFrame Columns:", X_data.head())# Check original columns
    missing_mask = np.zeros(X_data.shape, dtype=bool)
    if masking:
        found_base_col = set()
        for col in X_data.columns:
            base_col = col.rsplit(':', 1)[0]
            if base_col in found_base_col:
                continue
            found_base_col.add(base_col)
            non_missing_indices = X_data[col].dropna().index
            random_sample = np.random.choice(non_missing_indices, size=int(np.ceil(missingness_fraction * len(non_missing_indices))), replace=False)
            local_indices = X_data.index.get_indexer(random_sample)
            columns_with_same_base = [c for c in X_data.columns if c.startswith(base_col)]
            for col2 in columns_with_same_base:
                missing_mask[local_indices, X_data.columns.get_loc(col2)] = True
                X_data.loc[random_sample, col2] = np.nan
    else:
        for col in X_data.columns:
            non_missing_indices = X_data[col].dropna().index
            random_sample = np.random.choice(non_missing_indices, size=int(np.ceil(missingness_fraction * len(non_missing_indices))), replace=False)
            local_indices = X_data.index.get_indexer(random_sample)
            missing_mask[local_indices, X_data.columns.get_loc(col)] = True
            X_data.loc[random_sample, col] = np.nan
    return X_data, missing_mask

def generate_model_path(layers_config, autoencoder_builder, opt_config, act, fold, batch_size, masking_dir):
    """
    Generates the path for saving/loading model weights.
    
    Parameters:
    - layers_config: list of integers representing the architecture
    - autoencoder_builder: function used to build the model (e.g., build_ae)
    - opt_config: optimizer configuration dictionary
    - act: activation function name
    - fold: current fold number
    - batch_size: batch size used
    - masking_dir: path to the masking directory (with_masking or without_masking)

    Returns:
    - str: full file path to save/load the model
    """
    # converting layer config to folder name
    layer_folder = '_'.join(map(str, layers_config))

    # choosing subdirectory based on model type
    if autoencoder_builder == VAE_method:
        layers_subdir = os.path.join(masking_dir, 'vae_layers', layer_folder)
    else:
        layers_subdir = os.path.join(masking_dir, 'ae_dae_layers', layer_folder)

    # ensuring optimizer name is readable
    optimizer_name = opt_config['type'].__name__ if not isinstance(opt_config['type'], str) else opt_config['type']

    # final model file name
    model_filename = f"{autoencoder_builder.__name__}_{optimizer_name}_{act}_batch_{batch_size}_fold_{fold}.keras"

    # full path
    return os.path.join(layers_subdir, model_filename)

def initial_knn_imputed_data(X_train_scaled, X_val_with_missing, X_test_with_missing, fold, imputation_dir_knn):
    """
    Function to train or load KNN Imputer for each fold and save/load the imputed data.
    """
    imputer_name = f'KNN_Imputer_for_aes_Fold_{fold}'
    imputation_file = os.path.join(imputation_dir_knn, f"{imputer_name}.pkl")

    # paths for saving imputed validation and test data
    val_imputed_file = os.path.join(imputation_dir_knn, f"X_val_imputed_fold_{fold}.pkl")
    test_imputed_file = os.path.join(imputation_dir_knn, f"X_test_imputed_fold_{fold}.pkl")

    # if the imputed data already exists, loading it directly
    if os.path.exists(val_imputed_file) and os.path.exists(test_imputed_file):
        print(f"Loading imputed data for fold {fold}...")

        # loading the imputed validation data
        with open(val_imputed_file, 'rb') as f:
            X_val_imputed = pickle.load(f)

        # loading the imputed test data
        with open(test_imputed_file, 'rb') as f:
            X_test_imputed = pickle.load(f)
            
        print(f"Imputed data loaded for fold {fold}.")
        return X_val_imputed, X_test_imputed

    # if the imputer exists, load it
    if os.path.exists(imputation_file):
        with open(imputation_file, 'rb') as f:
            knn_imputer_fold = pickle.load(f)
        print(f"Loaded KNN imputer for fold {fold}.")
    else:
        # training the KNN imputer if not already saved
        print(f"Training KNN imputer for fold {fold}...")
        knn_imputer_fold = KNNImputer(n_neighbors=5)
        knn_imputer_fold.fit(X_train_scaled)
        
        # saving the trained KNN imputer
        with open(imputation_file, 'wb') as f:
            pickle.dump(knn_imputer_fold, f)
        print(f"KNN imputer trained and saved for fold {fold}.")

    # imputing validation and test data
    print(f"Imputing validation and test data for fold {fold}...")
    X_val_imputed = pd.DataFrame(knn_imputer_fold.transform(X_val_with_missing), 
                                 index=X_val_with_missing.index, columns=X_val_with_missing.columns)
    X_test_imputed = pd.DataFrame(knn_imputer_fold.transform(X_test_with_missing), 
                                  index=X_test_with_missing.index, columns=X_test_with_missing.columns)

    # saving the imputed validation and test data
    with open(val_imputed_file, 'wb') as f:
        pickle.dump(X_val_imputed, f)
    with open(test_imputed_file, 'wb') as f:
        pickle.dump(X_test_imputed, f)

    print(f"Imputed data saved for fold {fold}.")
    
    return X_val_imputed, X_test_imputed

def save_fold_data(fold, X_train, X_val, X_test, X_train_scaled, X_val_scaled, X_test_scaled, X_val_imputed, X_test_imputed, config, masking):
    '''
    Saves all input and imputed data for a given fold to avoid recomputation.
    '''
    # the masking directory correctly based on the `masking` parameter
    masking_dir = os.path.join(config['base_dir'], 'with_masking' if masking else 'without_masking')
    
    # full directory path for the fold
    fold_dir = os.path.join(masking_dir, 'for_aes_knn_imputed_fold', f'fold_{fold}')
    
    os.makedirs(fold_dir, exist_ok=True)
    
    fold_data_file = os.path.join(fold_dir, f"fold_data_{fold}.pkl")
    
    # loading the fold data if the file exists, otherwise save it
    if os.path.exists(fold_data_file):
        with open(fold_data_file, 'rb') as f:
            fold_data = pickle.load(f)
        print(f"Loaded existing data for fold {fold} from {fold_data_file}")
    else:
        fold_data = {
            'X_train' : X_train,
            'X_val' :  X_val,
            'X_test': X_test,
            
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'X_val_imputed': X_val_imputed,
            'X_test_imputed': X_test_imputed
        }
        with open(fold_data_file, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Saved new data for fold {fold} to {fold_data_file}")
    
    return fold_data

from tensorflow.keras import layers, models, regularizers, initializers, Input, Model
import tensorflow as tf

# shared configuration
kernel_initializer = initializers.he_uniform()
regularizer = regularizers.L2(0.001)
dropout_rate = 0.3

# AE (Model API, BatchNorm + Dropout)
def AE_method(input_dim, layers_config, activation):
    inputs = Input(shape=(input_dim,))
    x = inputs
    for units in layers_config:
        x = layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)
    return Model(inputs, outputs)

# DAE (same as AE but input assumed to be noisy)
def DAE_method(input_dim, layers_config, activation):
    inputs = Input(shape=(input_dim,))
    x = inputs
    for units in layers_config:
        x = layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    outputs = layers.Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)
    return Model(inputs, outputs)

# Sampling Layer for VAE
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE Custom Model
class VAEModel(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

# Final VAE Constructor
def VAE_method(input_dim, layers_config, activation):
    # Encoder
    inputs = Input(shape=(input_dim,))
    x = inputs
    for size in layers_config[:-1]:
        x = layers.Dense(size, activation=activation)(x)
    z_mean = layers.Dense(layers_config[-1])(x)
    z_log_var = layers.Dense(layers_config[-1])(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(layers_config[-1],))
    x = latent_inputs
    for size in reversed(layers_config[:-1]):
        x = layers.Dense(size, activation=activation)(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # Full VAE
    vae = VAEModel(encoder, decoder)
    vae.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])
    return vae

def run_models(X_train_scaled, X_val_imputed, X_test_imputed, X_val_scaled, X_train_scaled_noisy, 
               layers_configurations, vae_layers_configuration, optimizer_configs, 
               activations, X_test_scaled, test_surveys, fold, masking_dir, rows, initial_missingness, missingness_fraction, masking, 
               results_list, survey_r2_excel_list, column_metrics):

    '''
    Trains and evaluates AE, DAE, VAE models. Saves best models. Appends metrics to results list.
    '''

    for autoencoder_builder in [AE_method, DAE_method, VAE_method]:

        model_type = 'VAE' if autoencoder_builder == VAE_method else 'AE'
        layers_configs = vae_layers_configuration if model_type == 'VAE' else layers_configurations

        for batch_size in batch_sizes[model_type]:
            for opt_config in optimizer_configs:
                for act in activations:
                    for layers_config in layers_configs:

                        model_path = generate_model_path(layers_config, autoencoder_builder, opt_config, act, fold, batch_size, masking_dir)
                        imputed_data_path = model_path.replace('.keras', f'_imputed.pkl')

                        if os.path.exists(imputed_data_path):
                            print(f"Loading imputed values from {imputed_data_path}")
                            with open(imputed_data_path, 'rb') as f:
                                X_pred_df = pickle.load(f)
                        else:
                            model = autoencoder_builder(X_train_scaled.shape[1], layers_config, act)

                            if os.path.exists(model_path):
                                print(f"Loading model from {model_path}")
                                model.load_weights(model_path)
                            else:
                                print(f"Training: {autoencoder_builder.__name__}, Optimizer: {opt_config['type'].__name__}, Activation: {act}, Layers: {layers_config}, Batch size: {batch_size}")

                                #the model directory exists
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                                opt = opt_config['type'](learning_rate=opt_config['learning_rate'])
                                model.compile(optimizer=opt, loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])
                                checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                                early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True, verbose=1)

                                if autoencoder_builder == DAE_method:
                                    model.fit(X_train_scaled_noisy, X_train_scaled_noisy,
                                              epochs=1000, batch_size=batch_size, shuffle=True,
                                              validation_data=(X_val_imputed, X_val_imputed),
                                              callbacks=[checkpoint, early_stopping])
                                else:
                                    model.fit(X_train_scaled, X_train_scaled,
                                              epochs=1000, batch_size=batch_size, shuffle=True,
                                              validation_data=(X_val_imputed, X_val_imputed),
                                              callbacks=[checkpoint, early_stopping])

                                if os.path.exists(model_path):
                                    model.load_weights(model_path)
                                else:
                                    print(f"Warning: Training finished but model not saved at {model_path}")

                            # prediction
                            X_pred_scaled = model.predict(X_test_imputed)
                            X_pred_df = pd.DataFrame(X_pred_scaled, index=X_test_imputed.index, columns=X_test_imputed.columns)

                            with open(imputed_data_path, 'wb') as f:
                                pickle.dump(X_pred_df, f)
                            print(f"Imputed values saved to {imputed_data_path}")

                        # evaluation
                        imputer_name = 'KNN_initial_Imputer'
                        rmse = np.sqrt(mean_squared_error(X_test_scaled, X_pred_df))
                        r2 = r2_score(X_test_scaled, X_pred_df)
                        corr, _ = pearsonr(X_test_scaled.values.flatten(), X_pred_df.values.flatten())
                        available_data_fraction = 1 - missingness_fraction

                        results_list.append({
                            'Fold': fold,
                            'Actual Data': rows,
                            'Initial Missingness': initial_missingness,
                            'Introduced Missingness': missingness_fraction,
                            'Available Data': available_data_fraction,
                            'Train Shape': X_train_scaled.shape,
                            'Test Shape': X_test_scaled.shape,
                            'Validation Shape': X_val_scaled.shape,
                            'Optimizer': opt_config['type'].__name__,
                            'Activation': act,
                            'Batch_size': batch_size,
                            'Initial Imputer': imputer_name,
                            'Model': autoencoder_builder.__name__,
                            'Masking': 'with_masking' if masking else 'without_masking',
                            'Layers': str(layers_config),
                            'RMSE': rmse,
                            'R2': r2,
                            'Correlation': corr
                        })

                        for i in np.unique(test_surveys):
                            survey_mask = (test_surveys == i).values
                            X_test_survey_scaled = X_test_scaled[survey_mask]
                            X_pred_survey_scaled = X_pred_df[survey_mask]

                            if len(X_test_survey_scaled) > 1:
                                r2_survey = r2_score(X_test_survey_scaled, X_pred_survey_scaled)
                                rmse_survey = np.sqrt(mean_squared_error(X_test_survey_scaled, X_pred_survey_scaled))
                                corr_p_survey, _ = pearsonr(X_test_survey_scaled.values.flatten(), X_pred_survey_scaled.values.flatten())

                                survey_r2_excel_list.append({
                                    'Survey ID': i,
                                    'Fold': fold,
                                    'Initial Imputer': imputer_name,
                                    'Model': autoencoder_builder.__name__,
                                    'Masking': 'with_masking' if masking else 'without_masking',
                                    'Layer Config': str(layers_config),
                                    'RMSE': rmse_survey,
                                    'R2 ': r2_survey,
                                    'Correlation': corr_p_survey
                                })

                        for col in X_test_scaled.columns:
                            rmse_col = np.sqrt(mean_squared_error(X_test_scaled[col], X_pred_df[col]))
                            r2_col = r2_score(X_test_scaled[col], X_pred_df[col])
                            corr_col, _ = pearsonr(X_test_scaled[col], X_pred_df[col])
                            column_metrics.append({
                                'Column': col,
                                'Fold': fold,
                                'Optimizer': opt_config['type'].__name__,
                                'Activation': act,
                                'Batch_size': batch_size,
                                'Model': autoencoder_builder.__name__,
                                'Masking': 'with_masking' if masking else 'without_masking',
                                'Layers': str(layers_config),
                                'RMSE': rmse_col,
                                'R2': r2_col,
                                'Correlation': corr_col
                            })

    return rmse, r2, corr

def run_baseline_imputations(X_train_scaled, X_test_with_missing, X_val_with_missing, X_test_scaled, X_val_scaled, test_surveys, fold, masking, rows, 
                             initial_missingness, missingness_fraction, imputation_dir_knn, results_list, survey_r2_excel_list, column_metrics):

    '''
    Executes baseline imputers (Mean, KNN, MICE). Saves imputed data and computes evaluation metrics.
    '''
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'KNN_Base_Imputer': KNNImputer(n_neighbors=5),
        'MICE_Bayesian': IterativeImputer(estimator=BayesianRidge(), max_iter=1, random_state=6688),
        'MICE_Ridge': IterativeImputer(estimator=Ridge(), max_iter=1, random_state=6688)
    }
    
    for imputer_name_single, imputer in imputers.items():
        # path for the imputed data and the imputer model
        imputer_file = os.path.join(imputation_dir_knn, f"{imputer_name_single.replace(' ', '_')}_fold_{fold}.pkl")
        imputed_data_path = imputer_file.replace('.pkl', '_imputed.pkl')
        
        # checking if the imputed data already exists
        if os.path.exists(imputed_data_path):
            print(f"Loading imputed data from {imputed_data_path}")
            with open(imputed_data_path, 'rb') as f:
                X_test_imputed_df_single = pickle.load(f)
        else:
            # checking if the imputer model exists
            if os.path.exists(imputer_file):
                with open(imputer_file, 'rb') as f:
                    trained_imputer = pickle.load(f)
                print(f"Loaded {imputer_name_single} for fold {fold}.")
            else:
                # training the imputer and save it
                trained_imputer = imputer.fit(X_train_scaled)
                with open(imputer_file, 'wb') as f:
                    pickle.dump(trained_imputer, f)
                print(f"Trained and saved {imputer_name_single} for fold {fold}.")
            
            # imputing the test data
            X_test_imputed_single = trained_imputer.transform(X_test_with_missing)
            X_test_imputed_df_single = pd.DataFrame(X_test_imputed_single, index=X_test_with_missing.index, columns=X_test_with_missing.columns)
            
            # saving the imputed data using pickle
            with open(imputed_data_path, 'wb') as f:
                pickle.dump(X_test_imputed_df_single, f)
            print(f"Imputed data saved to {imputed_data_path}")
        
        # overall metrics
        rmse_single = np.sqrt(mean_squared_error(X_test_scaled, X_test_imputed_df_single))
        r2_single = r2_score(X_test_scaled, X_test_imputed_df_single)
        corr_single, _ = pearsonr(X_test_scaled.values.flatten(), X_test_imputed_df_single.values.flatten())
        
        # data availability info
        available_data_fraction = 1 - missingness_fraction
        
        # storing overall results
        results_list.append({
                            'Fold': fold,
                            'Actual Data': rows,
                            'Initial Missingness': initial_missingness,
                            'Introduced Missingness': missingness_fraction,
                            'Available Data': available_data_fraction,
                            'Train Shape': X_train_scaled.shape,
                            'Test Shape': X_test_scaled.shape,
                            'Validation Shape': X_val_scaled.shape,
                            'Optimizer': 'N/A',
                            'Activation': 'N/A',
                            'Batch_size': 'N/A',
                            'Initial Imputer': 'N/A',
                            'Model': imputer_name_single,
                            'Masking': 'with_masking' if masking else 'without_masking',
                            'Layers': 'N/A',
                            'RMSE': rmse_single,
                            'R2': r2_single,
                            'Correlation': corr_single
        })
        
        # storing metrics per survey
        for i in np.unique(test_surveys):
            survey_mask = (test_surveys == i).values 
            
            X_test_survey_scaled = X_test_scaled[survey_mask]
            X_test_imputed_survey_single = X_test_imputed_df_single[survey_mask]
            
            if len(X_test_survey_scaled) > 1:
                r2_survey_single = r2_score(X_test_survey_scaled, X_test_imputed_survey_single)
                rmse_survey_single = np.sqrt(mean_squared_error(X_test_survey_scaled, X_test_imputed_survey_single))
                corr_p_survey_single, _ = pearsonr(X_test_survey_scaled.values.flatten(), X_test_imputed_survey_single.values.flatten())
                
                survey_r2_excel_list.append({
                            'Survey ID': i,
                            'Fold': fold,
                            'Initial Imputer': 'N/A',
                            'Model': imputer_name_single,
                            'Masking': 'with_masking' if masking else 'without_masking',
                            'Layer Config': 'N/A',
                            'RMSE': rmse_survey_single,
                            'R2': r2_survey_single,
                            'Correlation': corr_p_survey_single
                })
            else:
                print(f"Not enough samples to calculate R2 and Correlation for survey {i} in fold {fold}")
        
        # storing metrics per column
        for col in X_test_scaled.columns:
            rmse_col_single = np.sqrt(mean_squared_error(X_test_scaled[col], X_test_imputed_df_single[col]))
            r2_col_single = r2_score(X_test_scaled[col], X_test_imputed_df_single[col])
            corr_col_single, _ = pearsonr(X_test_scaled[col], X_test_imputed_df_single[col])
            
            column_metrics.append({
                            'Column': col,
                            'Fold': fold,
                            'Optimizer': 'N/A',
                            'Activation': 'N/A',
                            'Batch_size': 'N/A',
                            'Model': imputer_name_single,
                            'Masking': 'with_masking' if masking else 'without_masking',
                            'Layers': 'N/A',
                            'RMSE': rmse_col_single,
                            'R2': r2_col_single,
                            'Correlation': corr_col_single
            })
    
    return rmse_single, r2_single, corr_single

def run_gain_method_v2(X_train_scaled, X_val_scaled, X_test_scaled, X_test_with_missing, gain_model_v2,
                       test_surveys, fold, results_list, survey_r2_excel_list, column_metrics, model_name,
                       rows, initial_missingness, missingness_fraction, masking, imputation_dir_gain_v2):
    '''
    Runs GAIN v2 with various epoch and batch size configurations, and logs evaluation metrics.
    '''

    configs = [
        {'epochs': 1, 'batch_size': 8192},
        {'epochs': 100, 'batch_size': 8192},
        {'epochs': 1, 'batch_size': 128},
        {'epochs': 100, 'batch_size': 128},
        {'epochs': 1, 'batch_size': 512},
        {'epochs': 100, 'batch_size': 512}
    ]

    for config in configs:
        epochs = config['epochs']
        batch_size = config['batch_size']
        tag = f"ep{epochs}_bs{batch_size}"

        gain_imputation_dir = os.path.join(imputation_dir_gain_v2, f"GAIN_v2_{tag}_fold_{fold}")
        os.makedirs(gain_imputation_dir, exist_ok=True)

        gain_imputed_file = os.path.join(gain_imputation_dir, f"gain_imputed_fold_{fold}.pkl")
        gain_imputed_csv_file = os.path.join(gain_imputation_dir, f"gain_imputed_fold_{fold}.csv")
        gain_model_file = os.path.join(gain_imputation_dir, f"gain_v2_model_fold_{fold}.weights.h5")

        gain_imputed = None

        if os.path.exists(gain_imputed_file):
            print(f"[{tag}] Imputed data already exists for fold {fold}. Loading...")
            with open(gain_imputed_file, 'rb') as f:
                gain_imputed = pickle.load(f)
        elif os.path.exists(gain_imputed_csv_file):
            print(f"[{tag}] CSV exists for fold {fold}. Loading...")
            gain_imputed_df = pd.read_csv(gain_imputed_csv_file, index_col=0)
        else:
            if not os.path.exists(gain_model_file):
                print(f"[{tag}] Training GAIN_v2 model for fold {fold}...")
                gain_model_v2.reinitialize()
                gain_model_v2.train(X_train_scaled.values, batch_size=batch_size, epochs=epochs)
                gain_model_v2.G.save_weights(gain_model_file)
                print(f"[{tag}] Model saved at {gain_model_file}")
            else:
                print(f"[{tag}] Loading model from {gain_model_file}")
                gain_model_v2.G.load_weights(gain_model_file)

            print(f"[{tag}] Running imputation...")
            gain_imputed = gain_model_v2.impute(X_test_with_missing.values)

            if np.isnan(gain_imputed).any():
                print(f"[{tag}] WARNING: NaNs in imputed values. Skipping.")
                gain_imputed = X_test_scaled.values
            else:
                with open(gain_imputed_file, 'wb') as f:
                    pickle.dump(gain_imputed, f)
                gain_imputed_df = pd.DataFrame(gain_imputed, index=X_test_scaled.index, columns=X_test_scaled.columns)
                gain_imputed_df.to_csv(gain_imputed_csv_file)
                print(f"[{tag}] Imputed data saved.")

        if gain_imputed is None:
            raise ValueError(f"[{tag}] GAIN imputation failed for fold {fold}.")

        gain_imputed_df = pd.DataFrame(gain_imputed, index=X_test_scaled.index, columns=X_test_scaled.columns)

        # Overall metrics
        rmse = np.sqrt(mean_squared_error(X_test_scaled, gain_imputed_df))
        r2 = r2_score(X_test_scaled, gain_imputed_df)
        corr, _ = pearsonr(X_test_scaled.values.flatten(), gain_imputed_df.values.flatten())
        available_data_fraction = 1 - missingness_fraction

        results_list.append({
            'Fold': fold,
            'Actual Data': rows,
            'Initial Missingness': initial_missingness,
            'Introduced Missingness': missingness_fraction,
            'Available Data': available_data_fraction,
            'Train Shape': X_train_scaled.shape,
            'Test Shape': X_test_scaled.shape,
            'Validation Shape': X_val_scaled.shape,
            'Optimizer': 'Adam',
            'Activation': 'ELU',
            'Batch_size': batch_size,
            'Epochs': epochs,
            'Initial Imputer': 'N/A',
            'Model': f"{model_name}_{tag}",
            'Masking': 'with_masking' if masking else 'without_masking',
            'Layers': '[475, 855, 855, dim]',
            'RMSE': rmse,
            'R2': r2,
            'Correlation': corr
        })

        for i in np.unique(test_surveys):
            survey_mask = (test_surveys == i).values
            X_test_survey_scaled = X_test_scaled[survey_mask]
            X_pred_survey_scaled = gain_imputed_df[survey_mask]

            if len(X_test_survey_scaled) > 1:
                r2_survey = r2_score(X_test_survey_scaled, X_pred_survey_scaled)
                rmse_survey = np.sqrt(mean_squared_error(X_test_survey_scaled, X_pred_survey_scaled))
                corr_p_survey, _ = pearsonr(X_test_survey_scaled.values.flatten(), X_pred_survey_scaled.values.flatten())

                survey_r2_excel_list.append({
                    'Survey ID': i,
                    'Fold': fold,
                    'Initial Imputer': 'N/A',
                    'Model': f"{model_name}_{tag}",
                    'Masking': 'with_masking' if masking else 'without_masking',
                    'Layer Config': '[475, 855, 855, dim]',
                    'RMSE': rmse_survey,
                    'R2': r2_survey,
                    'Correlation': corr_p_survey
                })

        for col in range(X_test_scaled.shape[1]):
            rmse_col = np.sqrt(mean_squared_error(X_test_scaled.iloc[:, col], gain_imputed_df.iloc[:, col]))
            r2_col = r2_score(X_test_scaled.iloc[:, col], gain_imputed_df.iloc[:, col])
            corr_col, _ = pearsonr(X_test_scaled.iloc[:, col], gain_imputed_df.iloc[:, col])
            column_metrics.append({
                'Column': X_test_scaled.columns[col],
                'Fold': fold,
                'Optimizer': 'Adam',
                'Activation': 'ELU',
                'Batch_size': batch_size,
                'Epochs': epochs,
                'Model': f"{model_name}_{tag}",
                'Masking': 'with_masking' if masking else 'without_masking',
                'Layers': '[475, 855, 855, dim]',
                'RMSE': rmse_col,
                'R2': r2_col,
                'Correlation': corr_col
            })

    return rmse, r2, corr

def calculate_fold_statistics(X_test, all_folds_stats, fold):
    """
    Collects mean and std per feature for each fold to allow comparison.
    Calculate and append the descriptive statistics of a fold to the overall statistics dictionary.

    Args:
        X_test (DataFrame): The test set for the current fold.
        all_folds_stats (DataFrame or None): DataFrame to accumulate fold statistics. None if this is the first fold.
        fold (int): The current fold number.

    Returns:
        DataFrame: The updated DataFrame containing fold statistics.
    """
    # descriptive statistics for this fold and transpose to get features as rows
    fold_stats = X_test.describe().transpose()

    fold_stats[f'Mean ± Std (Fold {fold+1})'] = fold_stats.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )

    if all_folds_stats is None:
        all_folds_stats = pd.DataFrame({
            'Feature': fold_stats.index,
            f'Fold {fold+1}': fold_stats[f'Mean ± Std (Fold {fold+1})']
        })
    else:

        all_folds_stats[f'Fold {fold+1}'] = fold_stats[f'Mean ± Std (Fold {fold+1})']

    return all_folds_stats

def calculate_levene_anova_stats(folds_data):
    '''
    Performs Levene's and ANOVA tests for each feature between fold pairs.
    '''
    levene_results = {'Feature': []}
    anova_results = {'Feature': []}

    # the pairs of folds to compare
    pairs_to_compare = [
        (0,0), (0,1), (0,2), (0,3), (0,4), 
        (1,0), (1,1), (1, 2), (1,3), (1,4), 
        (2,0), (2,1), (2,2), (2, 3), (2,4), 
        (3,0), (3,1), (3,2), (3,3), (3,4),
        (4,0), (4,1), (4,2), (4,3), (4,4)
    ]

    # initializing the dictionary for storing results
    for pair in pairs_to_compare:
        fold1, fold2 = pair
        levene_results[f'Fold {fold1} vs Fold {fold2} Statistic'] = []
        levene_results[f'Fold {fold1} vs Fold {fold2} p-value'] = []
        anova_results[f'Fold {fold1} vs Fold {fold2} Statistic'] = []
        anova_results[f'Fold {fold1} vs Fold {fold2} p-value'] = []

    # Looping over each feature
    for feature in folds_data[0].columns:
        levene_results['Feature'].append(feature)
        anova_results['Feature'].append(feature)
        
        # performing Levene's Test and ANOVA for each pair of folds
        for fold1, fold2 in pairs_to_compare:
            data1 = folds_data[fold1][feature].dropna()
            data2 = folds_data[fold2][feature].dropna()

            if data1.size > 0 and data2.size > 0:
                # performing Levene's Test
                levene_stat, levene_p = levene(data1, data2)
                levene_results[f'Fold {fold1} vs Fold {fold2} Statistic'].append(levene_stat)
                levene_results[f'Fold {fold1} vs Fold {fold2} p-value'].append(levene_p)

                # performing ANOVA
                anova_stat, anova_p = f_oneway(data1, data2)
                anova_results[f'Fold {fold1} vs Fold {fold2} Statistic'].append(anova_stat)
                anova_results[f'Fold {fold1} vs Fold {fold2} p-value'].append(anova_p)
            else:
                # appending NaN if any of the fold data is empty
                levene_results[f'Fold {fold1} vs Fold {fold2} Statistic'].append(np.nan)
                levene_results[f'Fold {fold1} vs Fold {fold2} p-value'].append(np.nan)
                anova_results[f'Fold {fold1} vs Fold {fold2} Statistic'].append(np.nan)
                anova_results[f'Fold {fold1} vs Fold {fold2} p-value'].append(np.nan)

    # converting to dfs
    levene_df = pd.DataFrame(levene_results)
    anova_df = pd.DataFrame(anova_results)

    return levene_df, anova_df

def process_and_save_fold_statistics(results_dir, all_folds_stats, rmse_values, r2_values, correlation_values, 
                                     results_list, fold_info_list, survey_r2_excel_list, column_metrics, 
                                     masking, levene_df, anova_df):
    """
    Process and save fold statistics, generating separate results for masking and without masking.
    
    Args:
        results_dir (str): The directtory where results should be saved.
        all_folds_stats (dict): Dictionary containing all fold statistics.
        rmse_values, r2_values, correlation_values (list): Lists of RMSE, R2, and Correlation values.
        results_list (list): List containing the results.
        fold_info_list, survey_r2_excel_list, column_metrics (list): Lists containing fold info, survey R2 scores, and column metrics.
        masking (bool): Whether the current results are with masking or without masking.
    
    Returns:
        tuple: DataFrames of the results list, average metrics, fold info, survey R2 scores, and column metrics.

    """
    summary_stats_df = pd.DataFrame(all_folds_stats)
    mask_str = "with_masking" if masking else "without_masking"

    # saving the statistics to Excel
    excel_file  = os.path.join(results_dir, f"fold_statistics.xlsx")
    with pd.ExcelWriter(excel_file) as writer:
        summary_stats_df.to_excel(writer, sheet_name='Summary Stats')
        # the Levene's and ANOVA results if they are provided
        if levene_df is not None:
            levene_df.to_excel(writer, sheet_name='Levene Test', index=False)
        if anova_df is not None:
            anova_df.to_excel(writer, sheet_name='ANOVA Test', index=False)
    print(f"All statistics have been saved to {excel_file}")

    # average and standard deviation for RMSE, R2, and Correlation across folds
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    mean_corr = np.mean(correlation_values)
    std_corr = np.std(correlation_values)

    # group by to calculate average metrics 
    results_df = pd.DataFrame(results_list)
    average_metrics = results_df.groupby(['Model', 'Optimizer', 'Activation', 'Layers', 'Batch_size', 'Masking']).agg({
        'RMSE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'Correlation': ['mean', 'std']
    }).reset_index()

    # renaming columns for readability
    average_metrics.columns = ['Model', 'Optimizer', 'Activation', 'Layers', 'Batch_size', 'Masking',
                               'Average RMSE', 'RMSE StdDev', 'Average R2', 'R2 StdDev', 'Average Correlation', 'Corr StdDev']

    # the mean ± StdDev columns
    average_metrics['Mean RMSE ± StdDev'] = average_metrics.apply(
        lambda row: f"{row['Average RMSE']:.4f} ± {row['RMSE StdDev']:.4f}", axis=1)

    average_metrics['Mean R2 ± StdDev'] = average_metrics.apply(
        lambda row: f"{row['Average R2']:.4f} ± {row['R2 StdDev']:.4f}", axis=1)
    
    average_metrics['Mean Correlation ± StdDev'] = average_metrics.apply(
        lambda row: f"{row['Average Correlation']:.4f} ± {row['Corr StdDev']:.4f}", axis=1)

    # pivot table to get R2 and Correlation scores for each fold
    pivot_df = results_df.pivot_table(index=['Model', 'Optimizer', 'Activation', 'Layers', 'Batch_size', 'Masking'],
                                      columns='Fold', values=['RMSE', 'R2', 'Correlation']).reset_index()

    # flattening the multiIndex columns and rename them
    pivot_df.columns = ['Model', 'Optimizer', 'Activation', 'Layers', 'Batch_size', 'Masking'] + [
        f'Fold{col[1]}_rmse' if col[0] == 'RMSE' else f'Fold{col[1]}_r2' if col[0] == 'R2' else f'Fold{col[1]}_corr' for col in pivot_df.columns[6:]
    ]

    # merging the pivoted DataFrame with the average results
    average_results_with_folds = pd.merge(average_metrics, pivot_df, on=['Model', 'Optimizer', 'Activation', 'Layers', 'Batch_size', 'Masking'])

    # the results using tabulate
    print(tabulate(average_results_with_folds, headers='keys', tablefmt='psql', showindex=False))

    # the average results with folds to Excel
    average_results_with_folds.to_excel(os.path.join(results_dir, f"average_results_with_folds_{mask_str}.xlsx"), index=False)

    return pd.DataFrame(results_list), average_metrics, pd.DataFrame(fold_info_list), pd.DataFrame(survey_r2_excel_list), pd.DataFrame(column_metrics)

# def calculate_country_statistics(df, country_col='Meta; adm0_gaul'):
#     """
#     Calculate descriptive statistics for each country in the dataset.
    
#     Args:
#         df (DataFrame): The dataset containing country-level data.
#         country_col (str): The column name that contains country identifiers.

#     Returns:
#         DataFrame: A DataFrame containing descriptive statistics for each country.
#     """
#     country_stats = {}
    
#     for country, group in df.groupby(country_col):
#         stats = group.describe().transpose()
#         stats['Country'] = country
#         stats['Mean ± Std'] = stats.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)
#         country_stats[country] = stats

#     # Combine all country statistics into a single DataFrame
#     combined_stats = pd.concat(country_stats.values(), axis=0)
#     return combined_stats

# country_statistics = calculate_country_statistics(input_df)

# Optionally, save the statistics to a CSV file
# country_statistics.to_csv('country_statistics.csv')