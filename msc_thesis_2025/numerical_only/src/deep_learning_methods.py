#!/usr/bin/env python3
# File: src/deep_learning_methods.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import time
import pickle
import pandas as pd
import tensorflow as tf
from src.utils import *
from src.gain import GAIN_code_v2  
from src.config_loader import load_config
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping
from src.config_loader import get_final_imputations_dir
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.utils import register_keras_serializable

# ---------------------------------------------------------
# Autoencoder (AE) - Using Model API
# ---------------------------------------------------------

def final_ae(input_dim, activation='elu'):
    """
    Constructs an Autoencoder (AE) model with fixed layer sizes.

    Args:
        input_dim (int): Dimensionality of the input features.
        activation (str): Activation function to use in hidden layers. Defaults to 'elu'.

    Returns:
        autoencoder (tf.keras.Model): Compiled Autoencoder model.
    """
    # fixed input and output layer sizes [128, x, 128], where x is dynamic based on input_dim
    layers_config = [128, 96, 128]
    
    kernel_initializer = 'he_uniform'
    regularizer = tf.keras.regularizers.L2(0.001)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    # first layer
    x = tf.keras.layers.Dense(layers_config[0], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # hidden layer (dynamic middle layer)
    x = tf.keras.layers.Dense(layers_config[1], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # output layer
    x = tf.keras.layers.Dense(layers_config[2], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # final output layer
    outputs = tf.keras.layers.Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)
    autoencoder = tf.keras.models.Model(inputs, outputs)
    
    return autoencoder

# ---------------------------------------------------------
# Denoising Autoencoder (DAE) - Using Model API
# ---------------------------------------------------------

def final_dae(input_dim, activation='elu'):
    """
    Constructs a Denoising Autoencoder (DAE) model.

    Args:
        input_dim (int): Dimensionality of the input features.
        activation (str): Activation function to use in hidden layers. Defaults to 'elu'.

    Returns:
        autoencoder (tf.keras.Model): Compiled Denoising Autoencoder model.
    """
    # fixed input and output layer sizes [128, x, 128], where x is dynamic based on input_dim
    layers_config = [128, 96, 128]
    
    kernel_initializer = 'he_uniform'
    regularizer = tf.keras.regularizers.L2(0.001)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    # first layer
    x = tf.keras.layers.Dense(layers_config[0], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    # hidden layer (dynamic middle layer)
    x = tf.keras.layers.Dense(layers_config[1], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    # output layer
    x = tf.keras.layers.Dense(layers_config[2], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    
    # final output layer
    outputs = tf.keras.layers.Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)
    autoencoder = tf.keras.models.Model(inputs, outputs)
    
    return autoencoder

# ---------------------------------------------------------
# Custom Classes for Variational Autoencoder (VAE)
# ---------------------------------------------------------
# 1. Sampling Layer: Implements the reparameterization trick
# 2. VAEModel Class: Combines encoder, decoder and KL divergence loss
# ---------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    """
    Custom Sampling layer for Variational Autoencoder (VAE).

    Generates samples from the latent space using the reparameterization trick.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.utils.register_keras_serializable()
class VAEModel(tf.keras.models.Model):
    """
    Custom Variational Autoencoder (VAE) model.

    Combines encoder, decoder, and Kullback-Leibler (KL) divergence loss.
    """
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

    def get_config(self):
        # getting the config of the encoder and decoder
        encoder_config = self.encoder.get_config()
        decoder_config = self.decoder.get_config()
        base_config = super(VAEModel, self).get_config()
        return {**base_config, "encoder": encoder_config, "decoder": decoder_config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder_config = config.pop('encoder')
        decoder_config = config.pop('decoder')

        if encoder_config is None or decoder_config is None:
            raise ValueError("Missing encoder or decoder config in VAEModel configuration")

        # recreating the encoder and decoder from their configs
        encoder = tf.keras.models.Model.from_config(encoder_config, custom_objects=custom_objects)
        decoder = tf.keras.models.Model.from_config(decoder_config, custom_objects=custom_objects)

        # returning the VAE model
        return cls(encoder, decoder, **config)

# ---------------------------------------------------------
# Variational Autoencoder (VAE) - Using Model API
# ---------------------------------------------------------

def final_vae(input_dim, activation='elu'):
    """
    Constructs a Variational Autoencoder (VAE) model.

    Args:
        input_dim (int): Dimensionality of the input features.
        activation (str): Activation function to use in hidden layers. Defaults to 'elu'.

    Returns:
        vae (VAEModel): Compiled VAE model.
    """
    layers_config = [4096, 2048]  # best configuration

    # encoder
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for size in layers_config[:-1]:
        x = tf.keras.layers.Dense(size, activation=activation)(x)
    z_mean = tf.keras.layers.Dense(layers_config[-1])(x)
    z_log_var = tf.keras.layers.Dense(layers_config[-1])(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # decoder
    latent_inputs = tf.keras.Input(shape=(layers_config[-1],))
    x = latent_inputs
    for size in reversed(layers_config[:-1]):
        x = tf.keras.layers.Dense(size, activation=activation)(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

    # VAE model
    vae = VAEModel(encoder, decoder)
    vae.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])

    return vae

# ---------------------------------------------------------
# Running Autoencoder (AE) for Missing Data Imputation
# ---------------------------------------------------------

def run_autoencoder(X_train, X_test_imputed_final, X_val_imputed_final, final_methods_dir, fold, repeat, missing_ratio, config):
    """
    Function to run Autoencoder imputation, saving and loading the model, imputed data, and timing info.
    
    Parameters:
    - X_train: Training data without missing values.
    - X_test_imputed_final: Test data with missing values.
    - X_val_imputed_final: Validation data for validation during training.
    - final_methods_dir: Directory to save/load models, imputed data, and timing info.
    - fold: Fold number.
    - repeat: Repeat number.
    - missing_ratio: The missing data ratio.

    Returns:
    - imputed_data: Imputed data for the test set.
    - training_time: Time taken for model training.
    - test_time: Time taken for model testing.
    """
    masking_dir, final_methods_dir, task2_dir, task3_time_dir = get_final_imputations_dir(config)

    # paths for saving the model, imputed data, and timing info
    imputed_data_path = os.path.join(final_methods_dir, f'AE_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'AE_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.keras')
    train_timing_info_path = os.path.join(task3_time_dir, f'AE_train_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    test_timing_info_path = os.path.join(task3_time_dir, f'AE_test_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')

    # checking if imputed data, model and timing info already exist
    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model for AE from {imputed_data_path} and {model_weights_path}")

        # loading the imputed data
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        with open(train_timing_info_path, 'rb') as f:
            training_time = pickle.load(f)
        with open(test_timing_info_path, 'rb') as f:
            test_time = pickle.load(f)

        # loading the saved AE model
        ae = tf.keras.models.load_model(model_weights_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
        print(f"Model loaded from {model_weights_path}")

        return imputed_data, training_time, test_time

    # initializing the Autoencoder model
    input_dim = X_train.shape[1]
    ae = final_ae(input_dim)

    # compiling the Autoencoder model
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])
    # setting up early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True)
    # tracking training time
    start_train_time = time.time()
    # training the Autoencoder model
    ae.fit(X_train, X_train, epochs=1000, batch_size=32, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])
    training_time = time.time() - start_train_time

    # tracking test time
    start_test_time = time.time()
    # generating the imputed data using the trained model
    imputed_data = ae.predict(X_test_imputed_final)
    imputed_data= pd.DataFrame(imputed_data, columns=X_test_imputed_final.columns)
    print("imputed_data of ae shape", imputed_data.shape)
    test_time = time.time() - start_test_time

    # saving the imputed data for future use
    save_pickle(imputed_data_path, imputed_data)
    ae.save(model_weights_path)
    save_pickle(train_timing_info_path, training_time)
    save_pickle(test_timing_info_path, test_time)

    # clearing the session to free up memory
    tf.keras.backend.clear_session()

    return imputed_data, training_time, test_time

# ---------------------------------------------------------------
# Running Denoising Autoencoder (DAE) for Missing Data Imputation
# ---------------------------------------------------------------

def run_dae(X_train, X_test_imputed_final, X_val_imputed_final, final_methods_dir, fold, repeat, missing_ratio, config):
    """
    Function to run DAE imputation, saving and loading the model, imputed data, and timing info.

    Parameters:
    - X_train: Training data without missing values.
    - X_test_imputed_final: Test data with missing values.
    - X_val_imputed_final: Validation data for validation during training.
    - final_methods_dir: Directory to save/load models, imputed data, and timing info.
    - fold: Fold number.
    - repeat: Repeat number.
    - missing_ratio: The missing data ratio.

    Returns:
    - imputed_data: Imputed data for the test set.
    - training_time: Time taken for model training.
    - test_time: Time taken for model testing.
    """
    masking_dir, final_methods_dir, task2_dir, task3_time_dir = get_final_imputations_dir(config)

    imputed_data_path = os.path.join(final_methods_dir, f'DAE_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'DAE_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.keras')
    train_timing_info_path = os.path.join(task3_time_dir, f'DAE_train_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    test_timing_info_path = os.path.join(task3_time_dir, f'DAE_test_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')

    # if imputed data and timing info already exist, load and return them
    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model weights for VAE from {imputed_data_path} and {model_weights_path}")
        
        # loading the imputed data
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        with open(train_timing_info_path, 'rb') as f:
            training_time = pickle.load(f)
        with open(test_timing_info_path, 'rb') as f:
            test_time = pickle.load(f)
        
        # loading the saved DAE model
        dae = tf.keras.models.load_model(model_weights_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
        # print(f"Model loaded from {model_weights_path}")

        return imputed_data, training_time, test_time

    # initializing the DAE model
    input_dim = X_train.shape[1]
    dae = final_dae(input_dim)

    dae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True)

    start_train_time = time.time()
    dae.fit(X_train, X_train, epochs=1000, batch_size=128, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])
    training_time = time.time() - start_train_time

    start_test_time = time.time()
    imputed_data = dae.predict(X_test_imputed_final)
    imputed_data= pd.DataFrame(imputed_data, columns=X_test_imputed_final.columns)
    print("imputed_data of dae shape", imputed_data.shape)
    test_time = time.time() - start_test_time

    # saving the imputed data, model, and timing information
    save_pickle(imputed_data_path, imputed_data)
    dae.save(model_weights_path)
    save_pickle(train_timing_info_path, training_time)
    save_pickle(test_timing_info_path, test_time)

    return imputed_data, training_time, test_time

# -----------------------------------------------------------------
# Running Variational Autoencoder (VAE) for Missing Data Imputation
# -----------------------------------------------------------------

def run_vae(X_train, X_test_imputed_final, X_val_imputed_final, final_methods_dir, fold, repeat, missing_ratio, config):
    """
    Function to run VAE imputation. Saves the imputed data, model, and timing info, and loads them if they already exist.

    Parameters:
    - X_train: Training data without missing values.
    - X_test_imputed_final: Test data with missing values.
    - X_val_imputed_final: Validation data for validation during training.
    - final_methods_dir: Directory to save/load models, imputed data, and timing info.
    - fold: Fold number.
    - repeat: Repeat number.
    - missing_ratio: The missing data ratio.

    Returns:
    - imputed_data: Imputed data for the test set.
    - training_time: Time taken for model training.
    - test_time: Time taken for model testing.
    """
    masking_dir, final_methods_dir, task2_dir, task3_time_dir = get_final_imputations_dir(config)

    imputed_data_path = os.path.join(final_methods_dir, f'VAE_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'VAE_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.keras')
    train_timing_info_path = os.path.join(task3_time_dir, f'VAE_train_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    test_timing_info_path = os.path.join(task3_time_dir, f'VAE_test_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')

    # checking if imputed data and model already exist
    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model weights for VAE from {imputed_data_path} and {model_weights_path}")
        
        # loading the imputed data
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        with open(train_timing_info_path, 'rb') as f:
            training_time = pickle.load(f)
        with open(test_timing_info_path, 'rb') as f:
            test_time = pickle.load(f)
        
        # loading the saved model with custom objects
        with custom_object_scope({'VAEModel': VAEModel, 'Sampling': Sampling, 'root_mean_squared_error': root_mean_squared_error}):
            vae = tf.keras.models.load_model(model_weights_path)
            # print(f"Model loaded from {model_weights_path}")

        return imputed_data, training_time, test_time

    # creating and train the VAE model
    input_dim = X_train.shape[1]
    vae = final_vae(input_dim)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)

    start_train_time = time.time()
    vae.fit(X_train, X_train, epochs=1000, batch_size=4096, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])
    training_time = time.time() - start_train_time

    start_test_time = time.time()
    imputed_data = vae.predict(X_test_imputed_final)
    imputed_data= pd.DataFrame(imputed_data, columns=X_test_imputed_final.columns)
    print("imputed_data of vae shape", imputed_data.shape)
    test_time = time.time() - start_test_time

    # saving the imputed data, model and timing information
    save_pickle(imputed_data_path, imputed_data)
    vae.save(model_weights_path)
    
    save_pickle(train_timing_info_path, training_time)
    save_pickle(test_timing_info_path, test_time)

    return imputed_data, training_time, test_time

# ------------------------------------------------------------------
# Running GAIN for Missing Data Imputation for numerical DHS dataset
# ------------------------------------------------------------------

def run_gain(X_train, X_test_with_missing, X_val_with_missing, final_methods_dir, fold, repeat, missing_ratio, config):
    """
    Function to run GAIN imputation. Saves the imputed data and timing info, and loads it if it already exists.
    
    Parameters:
    - X_train: Training data with no missing values.
    - X_test_imputed_final: Test data with missing values.
    - final_methods_dir: Directory where the imputed data will be saved.
    - fold: Current fold number.
    - repeat: Current repeat number.
    - missing_ratio: Ratio of missing data.
    
    Returns:
    - imputed_data: The imputed data after running GAIN.
    - training_time: The time taken to train the model.
    - test_time: The time taken to test (impute) the data.
    """
    masking_dir, final_methods_dir, task2_dir, task3_time_dir = get_final_imputations_dir(config)

    imputed_data_path = os.path.join(final_methods_dir, f'GAIN_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    gain_model_file = os.path.join(final_methods_dir, f'gain_v2_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.weights.h5')
    train_timing_info_path = os.path.join(task3_time_dir, f'GAIN_train_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    test_timing_info_path = os.path.join(task3_time_dir, f'GAIN_test_timing_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')

    gain_imputed = None

     # checking if imputed data already exists
    if os.path.exists(imputed_data_path):
        # print(f"Loading saved imputed data for GAIN from {imputed_data_path}")
        with open(imputed_data_path, 'rb') as f:
            gain_imputed = pickle.load(f)
        with open(train_timing_info_path, 'rb') as f:
            training_time = pickle.load(f)
        with open(test_timing_info_path, 'rb') as f:
            test_time = pickle.load(f)
            
        return gain_imputed, training_time, test_time

    else:
        # defining the dimension based on input training data
        dim = X_train.shape[1]
        gain_model_v2 = GAIN_code_v2(dim=dim)
    
        # tracking training time
        start_train_time = time.time()
    
        # settingup early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1)
    
        # training the model if weights do not exist
        if not os.path.exists(gain_model_file):
            print(f"Training GAIN_v2 model for fold {fold}...")
            gain_model_v2.reinitialize()
            gain_model_v2.train(X_train.values, batch_size=8192, epochs=1)
    
            # saving the trained model weights
            gain_model_v2.G.save_weights(gain_model_file)
            print(f"GAIN_v2 model saved for fold {fold} at {gain_model_file}")
        else:
            # loading the model weights if already trained
            print(f"Loading GAIN_v2 model weights from {gain_model_file}")
            gain_model_v2.G.load_weights(gain_model_file)
    
        training_time = time.time() - start_train_time
    
        # tracking test time
        start_test_time = time.time()
        gain_imputed = gain_model_v2.impute(X_test_with_missing.values)
        gain_imputed = pd.DataFrame(gain_imputed, columns=X_test_with_missing.columns)
        test_time = time.time() - start_test_time
    
        # checking for NaN values in the imputed df
        if gain_imputed.isnull().any().any():  # checking if any NaN values exist in the df
            print(f"WARNING: GAIN_v2 imputation resulted in NaN values for fold {fold}.")
        
        else:
            # saving the imputed df
            with open(imputed_data_path, 'wb') as f:
                pickle.dump(gain_imputed, f)
                print("imputed_data of gain shape", gain_imputed.shape)

        # saving timing information
        save_pickle(train_timing_info_path, training_time)
        save_pickle(test_timing_info_path, test_time)
        
    # raise an error if imputation failed
    if gain_imputed is None:
        raise ValueError(f"GAIN imputation failed for fold {fold}.")

    tf.keras.backend.clear_session()
    return gain_imputed, training_time, test_time

# Autoencoder model with fixed input/output layer configuration and dynamic middle layer (using Model API for AE, DAE, and VAE)
# ---------------------------------------------------------
# Autoencoder (AE) for Feature Evaluation with Model API
# ---------------------------------------------------------

def final_ae_feature_evaluation(input_dim, activation='elu'):
    # fixed input and output layer sizes [128, x, 128], where x is dynamic based on input_dim
    layers_config = [128, input_dim, 128]
    
    kernel_initializer = 'he_uniform'
    regularizer = tf.keras.regularizers.L2(0.001)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    # first layer
    x = tf.keras.layers.Dense(layers_config[0], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # hidden layer (dynamic middle layer)
    x = tf.keras.layers.Dense(layers_config[1], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # output layer
    x = tf.keras.layers.Dense(layers_config[2], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # final output layer
    outputs = tf.keras.layers.Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)
    autoencoder = tf.keras.models.Model(inputs, outputs)
    
    return autoencoder

# -----------------------------------------------------------------
# Denoising Autoencoder (DAE) for Feature Evaluation with Model API
# -----------------------------------------------------------------

# denoising Autoencoder (DAE) with Model API
def final_dae_feature_evaluation(input_dim, activation='elu'):
    # fixed input and output layer sizes [128, x, 128], where x is dynamic based on input_dim
    layers_config = [128, input_dim, 128]
    
    kernel_initializer = 'he_uniform'
    regularizer = tf.keras.regularizers.L2(0.001)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    # first layer
    x = tf.keras.layers.Dense(layers_config[0], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    
    # hidden layer (dynamic middle layer)
    x = tf.keras.layers.Dense(layers_config[1], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    
    # output layer
    x = tf.keras.layers.Dense(layers_config[2], activation=activation, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    
    # final output layer
    outputs = tf.keras.layers.Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)
    autoencoder = tf.keras.models.Model(inputs, outputs)
    
    return autoencoder

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.utils.register_keras_serializable()
class VAEModel(tf.keras.models.Model):
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

    def get_config(self):
        # getting the config of the encoder and decoder
        encoder_config = self.encoder.get_config()
        decoder_config = self.decoder.get_config()
        base_config = super(VAEModel, self).get_config()
        return {**base_config, "encoder": encoder_config, "decoder": decoder_config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder_config = config.pop('encoder')
        decoder_config = config.pop('decoder')

        if encoder_config is None or decoder_config is None:
            raise ValueError("Missing encoder or decoder config in VAEModel configuration")

        # recreating the encoder and decoder from their configs
        encoder = tf.keras.models.Model.from_config(encoder_config, custom_objects=custom_objects)
        decoder = tf.keras.models.Model.from_config(decoder_config, custom_objects=custom_objects)

        # returning the VAE model
        return cls(encoder, decoder, **config)

# -------------------------------------------------------------------
# Variational Autoencoder (VAE) for Feature Evaluation with Model API
# -------------------------------------------------------------------

# variational autoencoder (VAE) with Model API
def final_vae_feature_evaluation(input_dim, activation='elu'):
    # if input_dim <=30:
    #     layers_config = [4096, 2048] 
    # else:
    layers_config = [4096, round(21.33333* input_dim)]  # best configuration

    # encoder
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for size in layers_config[:-1]:
        x = tf.keras.layers.Dense(size, activation=activation)(x)
    z_mean = tf.keras.layers.Dense(layers_config[-1])(x)
    z_log_var = tf.keras.layers.Dense(layers_config[-1])(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # decoder
    latent_inputs = tf.keras.Input(shape=(layers_config[-1],))
    x = latent_inputs
    for size in reversed(layers_config[:-1]):
        x = tf.keras.layers.Dense(size, activation=activation)(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

    # VAE Model
    vae = VAEModel(encoder, decoder)
    vae.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])

    return vae

# ---------------------------------------------------------
# Run Autoencoder (AE) for Feature Evaluation
# ---------------------------------------------------------

# function to run Autoencoder with different numbers of features
def run_autoencoder_feature_evaluation(X_train, X_test_imputed_final, X_val_imputed_final, final_methods_dir, fold, repeat, n_features):
    """
    Function to run Autoencoder imputation, saving and loading the model and imputed data.
    
    Parameters:
    - X_train: Training data without missing values.
    - X_test_imputed_final: Test data with missing values.
    - X_val_imputed_final: Validation data for validation during training.
    - final_methods_dir: Directory to save/load models and imputed data.
    - fold: Fold number.
    - repeat: Repeat number.
    - n_features: Number of features used for evaluation.

    Returns:
    - imputed_data: Imputed data for the test set.
    """
    missing_ratio = 0.3  # fixed missing ratio
    imputed_data_path = os.path.join(final_methods_dir, f'AE_imputed_{n_features}_features_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'AE_model_{n_features}_features_fold_{fold}_repeat_{repeat}.keras')

    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model for AE from {imputed_data_path} and {model_weights_path}")
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        ae = tf.keras.models.load_model(model_weights_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
        print(f"Model loaded from {model_weights_path}")
        return imputed_data

    # initializing and train the Autoencoder
    input_dim = X_train.shape[1]
    
    ae = final_ae_feature_evaluation(input_dim)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True)
    
    ae.fit(X_train, X_train, epochs=1000, batch_size=32, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])
    imputed_data = ae.predict(X_test_imputed_final)

    # saving imputed data and model
    with open(imputed_data_path, 'wb') as f:
        pickle.dump(imputed_data, f)
    ae.save(model_weights_path)
    tf.keras.backend.clear_session()

    return imputed_data

# ---------------------------------------------------------
# Run Denoising Autoencoder (DAE) for Feature Evaluation
# ---------------------------------------------------------

# function to run DAE with different numbers of features
def run_dae_feature_evaluation(X_train, X_test_imputed_final, X_val_imputed_final, final_methods_dir, fold, repeat, n_features):
    missing_ratio = 0.3  # Fixed missing ratio
    imputed_data_path = os.path.join(final_methods_dir, f'DAE_imputed_{n_features}_features_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'DAE_model_{n_features}_features_fold_{fold}_repeat_{repeat}.keras')

    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model for DAE from {imputed_data_path} and {model_weights_path}")
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        dae = tf.keras.models.load_model(model_weights_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
        print(f"Model loaded from {model_weights_path}")
        return imputed_data

    # initializing and train the DAE
    input_dim = X_train.shape[1]

    def get_batch_size_for_dae(input_dim):
        if input_dim <= 15:
            return 64
        elif input_dim <= 30:
            return 64
        elif input_dim <= 45:
            return 64 #512
        elif input_dim <= 60:
            return 128 #2048
        elif input_dim <= 75:
            return 128
        else:
            return 128

    batch_size=get_batch_size_for_dae(input_dim)
    
    dae = final_dae_feature_evaluation(input_dim)
    dae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True)
    
    dae.fit(X_train, X_train, epochs=1000, batch_size=batch_size, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])
    imputed_data = dae.predict(X_test_imputed_final)

    # Save imputed data and model
    with open(imputed_data_path, 'wb') as f:
        pickle.dump(imputed_data, f)
    dae.save(model_weights_path)
    tf.keras.backend.clear_session()

    return imputed_data

# ---------------------------------------------------------
# Run Variational Autoencoder (VAE) for Feature Evaluation
# ---------------------------------------------------------

def run_vae_feature_evaluation(X_train, X_test_imputed_final, X_val_imputed_final, final_methods_dir, fold, repeat, n_features):
    """
    Function to run VAE imputation. Saves the imputed data and model, and loads them if they already exist.

    Parameters:
    - X_train: Training data.
    - X_test_imputed_final: Test data with missing values imputed.
    - X_val_imputed_final: Validation data for imputation.
    - final_methods_dir: Directory to save/load models and imputed data.
    - fold: Fold number.
    - repeat: Repeat number.
    - n_features: Number of selected features for this evaluation.

    Returns:
    - imputed_data: Imputed data for the test set.
    """
    # path to saved imputed data and model
    imputed_data_path = os.path.join(final_methods_dir, f'VAE_imputed_{n_features}_features_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'VAE_model_{n_features}_features_fold_{fold}_repeat_{repeat}.keras')

    # checking if imputed data and model already exist
    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model weights for VAE from {imputed_data_path} and {model_weights_path}")
        
        # loading the imputed data
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        
        # loading the saved model with custom objects
        with custom_object_scope({'VAEModel': VAEModel, 'Sampling': Sampling, 'root_mean_squared_error': root_mean_squared_error}):
            vae = tf.keras.models.load_model(model_weights_path)
            print(f"Model loaded from {model_weights_path}")

        return imputed_data

    # if no saved model exists, create and train the VAE model
    input_dim = X_train.shape[1]

    def get_batch_size(input_dim):
        if input_dim <= 15:
            return 4096
        elif input_dim <= 30:
            return 4096
        elif input_dim <= 45:
            return 4096 #512
        elif input_dim <= 60:
            return 4096 #2048
        elif input_dim <= 75:
            return 4096
        else:
            return 4096

    batch_size=get_batch_size(input_dim)
    
    vae = final_vae_feature_evaluation(input_dim)

    # setting up early stopping callback
    # patience = 35 if input_dim <= 30 else 10
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

    # training the VAE model
    vae.fit(X_train, X_train, 
            epochs=1000, 
            batch_size=batch_size, 
            shuffle=True, 
            validation_data=(X_val_imputed_final, X_val_imputed_final), 
            callbacks=[early_stopping])

    # generating the imputed data using the trained model
    imputed_data = vae.predict(X_test_imputed_final)

    # saving the imputed data for future use
    with open(imputed_data_path, 'wb') as f:
        pickle.dump(imputed_data, f)
        print(f"Imputed data saved to {imputed_data_path}")

    # saving the trained VAE model
    vae.save(model_weights_path)
    print(f"Model saved to {model_weights_path}")

    tf.keras.backend.clear_session()

    return imputed_data

# ---------------------------------------------------------
# Run GAIN for Feature Evaluation
# ---------------------------------------------------------

def run_gain_feature_evaluation(X_train, X_test_with_missing, X_val_with_missing, final_methods_dir, fold, repeat, n_features):
    """
    Function to run GAIN imputation. Saves the imputed data and loads it if it already exists.
    
    Parameters:
    - X_train: Training data with no missing values.
    - X_test_with_missing: Test data with missing values.
    - X_val_with_missing: Validation data with missing values.
    - final_methods_dir: Directory where the imputed data will be saved.
    - fold: Current fold number.
    - repeat: Current repeat number.
    - n_features: Number of selected features for this evaluation.
    
    Returns:
    - imputed_data: The imputed data after running GAIN.
    """
    
    imputed_data_path = os.path.join(final_methods_dir, f'GAIN_imputed_{n_features}_features_fold_{fold}_repeat_{repeat}.pkl')
    gain_model_file = os.path.join(final_methods_dir, f'gain_v2_model_{n_features}_features_fold_{fold}_repeat_{repeat}.weights.h5')

    gain_imputed = None

    # checking if imputed data already exists
    if os.path.exists(imputed_data_path):
        print(f"Loading saved imputed data for GAIN from {imputed_data_path}")
        with open(imputed_data_path, 'rb') as f:
            gain_imputed = pickle.load(f)
        return gain_imputed

    else:
        # defining the dimension based on input training data
        dim = X_train.shape[1]
        gain_model_v2 = GAIN_code_v2(dim=dim)
    
        # setting up early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',   
            patience=35,          
            restore_best_weights=True,  
            verbose=1
        )
    
        # training the model if weights don't exist
        if not os.path.exists(gain_model_file):
            print(f"Training GAIN_v2 model for fold {fold} with {n_features} features...")
            gain_model_v2.reinitialize()
                    
            gain_model_v2.train(X_train.values, batch_size=8192, epochs=1)
            
            # saving the trained model weights
            gain_model_v2.G.save_weights(gain_model_file)
            print(f"GAIN_v2 model saved for fold {fold} at {gain_model_file}")
        else:
            # loading the model weights if already trained
            print(f"Loading GAIN_v2 model weights from {gain_model_file}")
            gain_model_v2.G.load_weights(gain_model_file)
            
        print(f"Running GAIN_v2 imputation for fold {fold} with {n_features} features using the trained model...")
        gain_imputed = gain_model_v2.impute(X_test_with_missing.values)
        
        # checking for NaN values in the imputed data
        if np.isnan(gain_imputed).any():
            print(f"WARNING: GAIN_v2 imputation resulted in NaN values for fold {fold}. Skipping this imputation.")
            gain_imputed = X_test_with_missing.values  # filling with original data where NaNs are present
        else:
            # saving the imputed data as a pickle file
            with open(imputed_data_path, 'wb') as f:
                pickle.dump(gain_imputed, f)
            print(f"Imputed data saved for GAIN with {n_features} features.")

    tf.keras.backend.clear_session()

    return gain_imputed