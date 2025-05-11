#!/usr/bin/env python3
# File: src/imputations.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# importing required libraries
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from src.gain import GAIN_code_v2
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from tensorflow.keras.losses import mse
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import custom_object_scope
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.config import RESULTS_DIR, GET_FINAL_IMPUTATIONS_DIR
from src.preprocessing import one_hot_encode_for_others, consistent_one_hot_encode
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Add, GaussianNoise
from src.utils import root_mean_squared_error, convert_sparse_to_dense, save_pickle, adapt_original_mask_to_one_hot

# -------------------------------------------------------------------------------------------------------
# Non-deep learning (statistical and traditional ML) methods for 'numerical with categorical' DHS Dataset
# -------------------------------------------------------------------------------------------------------

def process_single_imputation(train_data_with_missing, test_data, test_data_with_missing, missing_ratio, fold, repeat, output_dir, original_categorical_mask):
    """
    Performs or loads imputed data using Mean, KNN and MICE_Ridge imputation methods.

    Parameters:
    - train_data_with_missing: Training dataset with missing values.
    - test_data: Complete test dataset without missing values (used as reference structure).
    - test_data_with_missing: Test dataset with missing values.
    - missing_ratio: Proportion of missing data introduced into datasets.
    - fold: Cross-validation fold identifier.
    - repeat: Experiment repetition number.
    - output_dir: Directory to save or load imputed datasets and timings.
    - original_categorical_mask: Original mask indicating positions of categorical missingness.

    Returns:
    - imputed_data_store: Dictionary of imputed test datasets for each imputation method.
    """
    # imputation methods to evaluate
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'KNN': KNNImputer(n_neighbors=5),
        'MICE_Ridge': IterativeImputer(estimator=Ridge(), max_iter=10, random_state=6688)
    }

    # ensuring test_data_with_missing is a DataFrame with correct structure
    test_data_with_missing = pd.DataFrame(test_data_with_missing, columns=test_data.columns, index=test_data.index)

    # converting sparse columns to dense using dtype check
    test_data_with_missing = test_data_with_missing.apply(lambda col: col.sparse.to_dense() if isinstance(col.dtype, pd.SparseDtype) else col)

    # ensuring column alignment between train and test
    train_cols = train_data_with_missing.columns
    test_cols = test_data_with_missing.columns

    # identifying missing and extra columns between datasets
    missing_cols = list(set(train_cols) - set(test_cols))
    extra_cols = list(set(test_cols) - set(train_cols))

    # handling missing columns by filling with zeros (default strategy)
    if missing_cols:
        fill_df = pd.DataFrame(0, index=test_data_with_missing.index, columns=missing_cols)
        test_data_with_missing = pd.concat([test_data_with_missing, fill_df], axis=1)

    # dropping extra columns not present in the training dataset
    if extra_cols:
        test_data_with_missing.drop(columns=extra_cols, inplace=True, errors='ignore')

    # ensuring column order matches exactly with training data
    test_data_with_missing = test_data_with_missing[train_cols]  # ensuring column order

    # dictionaries to store results and timing information
    imputed_data_store = {}

    # looping through each imputation method
    for imputer_name, imputer in imputers.items():
        # the path for saved imputed data
        imputed_data_path = os.path.join(output_dir, f"{imputer_name}_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")

        # checking if previously imputed data exists
        if os.path.exists(imputed_data_path):
            # loading existing imputed dataset to save computation
            imputed_df = pd.read_pickle(imputed_data_path)
            print(f"{imputer_name} imputed data loaded with shape: {imputed_df.shape}")
        else:
            # training imputer on training data with missing values
            imputer.fit(train_data_with_missing)

            # imputing missing values on test dataset
            imputed_df = pd.DataFrame(
                imputer.transform(test_data_with_missing), 
                columns=test_data_with_missing.columns, 
                index=test_data_with_missing.index
            )

            print(f" {imputer_name} imputed data shape: {imputed_df.shape}")

            # saving imputed results to avoid future recomputation
            with open(imputed_data_path, 'wb') as f:
                pickle.dump(imputed_df, f)

        # ensuring valid df
        if isinstance(imputed_df, pd.DataFrame):
            imputed_data_store[imputer_name] = imputed_df
        else:
            raise ValueError(f"Imputed data for {imputer_name} is not a valid DataFrame!")

    return imputed_data_store

# ---------------------------------------------------------------------------------------
# Improved Autoencoder (AE) for 'numerical with categorical' DHS Dataset- Using Model API
# ---------------------------------------------------------------------------------------

def improved_ae(input_dim, activation='leaky_relu'):
    """
    Builds an improved Autoencoder model with deeper architecture,
    batch normalization, dropout and LeakyReLU activation.

    Parameters:
    - input_dim (int): Dimensionality of the input data.
    - activation (str): Activation function used throughout the network ('leaky_relu').

    Returns:
    - autoencoder: Compiled Keras autoencoder model.
    """
    # defining layer sizes for encoder and decoder
    layers_config = [512, 256, 128]  # deeper structure
    
    kernel_initializer = 'he_uniform'
    regularizer = tf.keras.regularizers.L2(0.0005)  # reducing weight decay
    dropout_rate = 0.2  # lowering dropout for better learning
    
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    # first Hidden Layer (Encoder)
    x = tf.keras.layers.Dense(layers_config[0], kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # second Hidden Layer (Bottleneck Layer)
    x = tf.keras.layers.Dense(layers_config[1], kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # expanding Layer
    # third Hidden Layer (Decoder)
    x = tf.keras.layers.Dense(layers_config[2], kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # output Layer with linear activation for regression tasks
    outputs = tf.keras.layers.Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)

    # defining and compiling the autoencoder model
    autoencoder = tf.keras.models.Model(inputs, outputs)

    # compiling model with accuracy as metric
    autoencoder.compile(
        optimizer=AdamW(learning_rate=0.001), # AdamW for better generalization
        loss='mse', # reconstruction loss
        metrics=['accuracy']  # accuracy metric for evaluation
    )
    return autoencoder

# --------------------------------------------------------------------------------------------------
# Improved Denoising Autoencoder (DAE) for 'numerical with categorical' DHS Dataset- Using Model API
# --------------------------------------------------------------------------------------------------

def improved_dae(input_dim, activation='leaky_relu'):
    """
    Builds an improved Denoising Autoencoder (DAE) model with Gaussian noise,
    skip connections, deeper layers, batch normalization, and LeakyReLU activation.

    Parameters:
    - input_dim (int): Dimensionality of the input data.
    - activation (str): Activation function used throughout the network ('leaky_relu').

    Returns:
    - dae (DAEModel): Compiled Keras DAE model.
    """
    # layer configuration for deeper denoising autoencoder
    layers_config = [1024, 512, 256, 121]  # deeper architecture
    kernel_initializer = 'he_uniform'
    regularizer = tf.keras.regularizers.L2(0.0001)  # L2 regularization 
    dropout_rate = 0.15  # reduce dropout slightly for better stability

    inputs = Input(shape=(input_dim,))
    # adding slight Gaussian noise to the inputs for denoising
    x = GaussianNoise(0.1)(inputs)  # reduced noise for stability

    # preserving original inputs for skip connection
    skip = x  # saving input for skip connection

    # encoder-decoder layers
    for size in layers_config:
        x = Dense(size, kernel_initializer=kernel_initializer, activity_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(dropout_rate)(x)

    # skipping connection to reinforce learning of original input features
    x = Add()([x, skip])

    # output layer for reconstruction
    outputs = Dense(input_dim, activation='linear', kernel_initializer=kernel_initializer)(x)

    # defining and compiling the denoising autoencoder model
    autoencoder = Model(inputs, outputs)

    # compiling model
    autoencoder.compile(
        optimizer=AdamW(learning_rate=0.0005),  # AdamW for better generalization
        loss='mse',  # reconstruction loss
        metrics=['accuracy']  # accuracy metric for evaluation
    )

    return autoencoder

# ---------------------------------------------------------
# Custom Classes for Variational Autoencoder (VAE)
# ---------------------------------------------------------
# 1. Sampling Layer: Implements the reparameterization trick
# 2. VAEModel Class: Combines encoder, decoder and KL divergence loss
# ---------------------------------------------------------

from tensorflow.keras.utils import custom_object_scope
# sampling layer using reparameterization trick
@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample latent vector z via reparameterization.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# custom VAE Model Class
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

# ----------------------------------------------------------------------------------------------------
# Improved Variational Autoencoder (VAE) for 'numerical with categorical' DHS Dataset- Using Model API
# ----------------------------------------------------------------------------------------------------

def improved_vae(input_dim, activation='relu'):

    """
    Constructs an improved Variational Autoencoder (VAE) with deeper layers and a larger latent dimension.

    Parameters:
    - input_dim (int): Dimensionality of the input data.
    - activation (str): Activation function used throughout the network ('leaky_relu').

    Returns:
    - vae (VAEModel): Compiled Keras VAE model.
    """
    # encoder configuration (deep structure with large latent space)
    #layers_config = [2048, 1024, 512]  # larger bottleneck
    layers_config = [4096, 2048]

    # encoder network
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for size in layers_config[:-1]:
        x = tf.keras.layers.Dense(size, activation=activation)(x)
        
    # latent variables (mean and log variance for sampling)
    z_mean = tf.keras.layers.Dense(layers_config[-1])(x)
    z_log_var = tf.keras.layers.Dense(layers_config[-1])(x)
    # sample latent vector
    z = Sampling()([z_mean, z_log_var])
    
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # decoder network
    latent_inputs = tf.keras.Input(shape=(layers_config[-1],))
    x = latent_inputs
    for size in reversed(layers_config[:-1]):
        x = tf.keras.layers.Dense(size, activation=activation)(x)

    # # output layer for reconstruction
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

    # instantiating and compiling the full VAE model
    vae = VAEModel(encoder, decoder)
    # vae.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

    vae.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss='mse',
        metrics=['accuracy']
    )
    
    return vae

# -------------------------------------------------------------
# Running Improved Autoencoder (AE) for Missing Data Imputation
# -------------------------------------------------------------

def run_autoencoder(X_train, X_val_imputed_final, X_test_imputed_final, final_methods_dir, fold, repeat, missing_ratio, original_categorical_mask):
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
    - original_categorical_mask: Mask for categorical columns in original data.

    Returns:
    - imputed_data: Imputed data for the test set.
    """

    # paths for saving the model, imputed data, and timing info
    imputed_data_path = os.path.join(final_methods_dir, f'AE_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'AE_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.keras')

    # converting SparseArrays in `X_test_imputed_final` to dense arrays
    if isinstance(X_test_imputed_final, pd.DataFrame):
        imputed_data = convert_sparse_to_dense(X_test_imputed_final)

    X_test_imputed_final= pd.DataFrame(X_test_imputed_final)

    # adapting the original mask to one-hot encoded structure
    one_hot_mask = adapt_original_mask_to_one_hot(X_test_imputed_final, original_categorical_mask)

    # checking if imputed data, model, and timing info already exist
    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model for AE from {imputed_data_path} and {model_weights_path}")

        # loading the imputed data
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)

        # Load the saved AE model
        ae = tf.keras.models.load_model(model_weights_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
        print(f"Model loaded from {model_weights_path}")

        return imputed_data

    # initializing the Autoencoder model
    input_dim = X_train.shape[1]
    ae = improved_ae(input_dim)

    # compiling the Autoencoder model
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])

    # setting up early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True)

    # training the Autoencoder model
    ae.fit(X_train, X_train, epochs=1000, batch_size=32, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])

    # generating the imputed data using the trained model
    imputed_data = ae.predict(X_test_imputed_final)
    # asserting imputed_data.shape[0] == X_test_imputed_final.shape[0], "🚨 Mismatch in number of rows!"
    imputed_data= pd.DataFrame(imputed_data, columns=X_test_imputed_final.columns, index=X_test_imputed_final.index)

    # saving the imputed data for future use
    save_pickle(imputed_data_path, imputed_data)
    ae.save(model_weights_path)

    # clearing the session to free up memory
    tf.keras.backend.clear_session()

    return imputed_data

# ------------------------------------------------------------------------
# Running Improved Denoising Autoencoder (DAE) for Missing Data Imputation
# ------------------------------------------------------------------------

def run_dae(X_train, X_val_imputed_final, X_test_imputed_final, final_methods_dir, fold, repeat, missing_ratio, original_categorical_mask):
    """
    Function to run DAE imputation, saving and loading the model, imputed data and timing info.

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
    """

    imputed_data_path = os.path.join(final_methods_dir, f'DAE_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'DAE_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.keras')

    # converting SparseArrays in `X_test_imputed_final` to dense arrays
    if isinstance(X_test_imputed_final, pd.DataFrame):
        imputed_data = convert_sparse_to_dense(X_test_imputed_final)

    X_test_imputed_final= pd.DataFrame(X_test_imputed_final)

    # adapting the original mask to one-hot encoded structure
    one_hot_mask = adapt_original_mask_to_one_hot(X_test_imputed_final, original_categorical_mask)

    # if imputed data and timing info already exist, load and return them
    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model weights for VAE from {imputed_data_path} and {model_weights_path}")
        
        # loading the imputed data
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        
        # loading the saved DAE model
        dae = tf.keras.models.load_model(model_weights_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
        # print(f"Model loaded from {model_weights_path}")

        return imputed_data

    # initializing the DAE model
    input_dim = X_train.shape[1]
    dae = improved_dae(input_dim)

    dae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error, metrics=[tf.keras.metrics.MeanSquaredError()])
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True)

    dae.fit(X_train, X_train, epochs=1000, batch_size=128, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])

    imputed_data = dae.predict(X_test_imputed_final)
    imputed_data= pd.DataFrame(imputed_data, columns=X_test_imputed_final.columns, index=X_test_imputed_final.index)

    # saving the imputed data, model, and timing information
    save_pickle(imputed_data_path, imputed_data)
    dae.save(model_weights_path)

    return imputed_data

# --------------------------------------------------------------------------
# Running Improved Variational Autoencoder (VAE) for Missing Data Imputation
# --------------------------------------------------------------------------

def run_vae(X_train, X_val_imputed_final, X_test_imputed_final, final_methods_dir, fold, repeat, missing_ratio, original_categorical_mask):
    """
    Function to run VAE imputation. Saves the model, imputed data and timing info and loads them if they already exist.
    """

    imputed_data_path = os.path.join(final_methods_dir, f'VAE_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    model_weights_path = os.path.join(final_methods_dir, f'VAE_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.keras')

    # converting SparseArrays in `X_test_imputed_final` to dense arrays
    if isinstance(X_test_imputed_final, pd.DataFrame):
        imputed_data = convert_sparse_to_dense(X_test_imputed_final)

    X_test_imputed_final= pd.DataFrame(X_test_imputed_final)

    if isinstance(imputed_data, pd.DataFrame):
        imputed_data = convert_sparse_to_dense(imputed_data)

    # adapting the original mask to one-hot encoded structure
    one_hot_mask = adapt_original_mask_to_one_hot(X_test_imputed_final, original_categorical_mask)

    # checking if imputed data and model already exist
    if os.path.exists(imputed_data_path) and os.path.exists(model_weights_path):
        print(f"Loading saved imputed data and model weights for VAE from {imputed_data_path} and {model_weights_path}")
        
        # loading the imputed data
        with open(imputed_data_path, 'rb') as f:
            imputed_data = pickle.load(f)
        
        # loading the saved model with custom objects
        with custom_object_scope({'VAEModel': VAEModel, 'Sampling': Sampling, 'root_mean_squared_error': root_mean_squared_error}):
            vae = tf.keras.models.load_model(model_weights_path)
            # print(f"Model loaded from {model_weights_path}")

        return imputed_data

    # creating and train the VAE model
    input_dim = X_train.shape[1]
    vae = improved_vae(input_dim)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    vae.fit(X_train, X_train, epochs=300, batch_size=4096, shuffle=True, validation_data=(X_val_imputed_final, X_val_imputed_final), callbacks=[early_stopping])

    imputed_data = vae.predict(X_test_imputed_final)
    imputed_data= pd.DataFrame(imputed_data, columns=X_test_imputed_final.columns, index=X_test_imputed_final.index)

    # saving the imputed data, model and timing information
    save_pickle(imputed_data_path, imputed_data)
    vae.save(model_weights_path)

    return imputed_data

# ---------------------------------------------------------
# Running GAIN for numerical+categorical DHS dataset
# ---------------------------------------------------------

def run_gain(X_train, X_test_with_missing, X_val_with_missing, test_indices, final_methods_dir, fold, repeat, missing_ratio, original_categorical_mask):
    """
    Function to run GAIN imputation. Saves the imputed data and timing info, and loads it if it already exists.
    
    Parameters:
    - X_train: Training data with no missing values.
    - X_test_with_missing: Test data with missing values (before filtering).
    - X_val_with_missing: Validation data with missing values.
    - test_indices: Indices corresponding only to the test dataset.
    - final_methods_dir: Directory where the imputed data will be saved.
    - fold: Current fold number.
    - repeat: Current repeat number.
    - missing_ratio: Ratio of missing data.
    
    Returns:
    - imputed_data: The imputed data after running GAIN.
    """

    imputed_data_path = os.path.join(final_methods_dir, f'GAIN_imputed_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl')
    gain_model_file = os.path.join(final_methods_dir, f'gain_v2_model_{missing_ratio}_fold_{fold}_repeat_{repeat}.weights.h5')

    gain_imputed = None

    # checking if imputed data already exists
    if os.path.exists(imputed_data_path):
        with open(imputed_data_path, 'rb') as f:
            gain_imputed = pickle.load(f)
        return gain_imputed

    else:
        # ensuring Only test data is selected
        X_test_with_missing = X_test_with_missing.loc[test_indices]  # Filter test data properly
        print(f" Filtered X_test_with_missing shape: {X_test_with_missing.shape}")

        # adapting the original mask to one-hot encoded structure
        one_hot_mask = adapt_original_mask_to_one_hot(X_test_with_missing, original_categorical_mask)

        # converting SparseArrays in `X_test_with_missing` to dense arrays
        if isinstance(X_test_with_missing, pd.DataFrame):
            imputed_data = convert_sparse_to_dense(X_test_with_missing)
    
        X_test_with_missing= pd.DataFrame(X_test_with_missing)

        # reintroducing NaNs in one-hot encoded columns based on the adapted mask
        X_test_with_missing[one_hot_mask] = np.nan

        # creating a categorical mask (1 if column is categorical, 0 if numerical)
        categorical_cols = X_train.columns[X_train.dtypes == 'uint8']  # or 'object' if not one-hot encoded
        cat_mask_array = np.array([1.0 if col in categorical_cols else 0.0 for col in X_train.columns], dtype=np.float32)

        # defining the dimension based on input training data
        dim = X_train.shape[1]
        gain_model_v2 = GAIN_code_v2(dim=dim, cat_mask=cat_mask_array)
        # gain_model_v2 = GAIN_code_v2(dim=dim)

        # setup early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1)

        # training the model if weights don't exist
        if not os.path.exists(gain_model_file):
            print(f"Training GAIN_v2 model for fold {fold}...")
            gain_model_v2.reinitialize()
            gain_model_v2.train(X_train.values, batch_size=8192, epochs=20)

            # saving the trained model weights
            gain_model_v2.G.save_weights(gain_model_file)
            print(f"GAIN_v2 model saved for fold {fold} at {gain_model_file}")
        else:
            # loading the model weights if already trained
            print(f"Loading GAIN_v2 model weights from {gain_model_file}")
            gain_model_v2.G.load_weights(gain_model_file)

        gain_imputed = gain_model_v2.impute(X_test_with_missing.values)
        gain_imputed = pd.DataFrame(gain_imputed, columns=X_test_with_missing.columns, index=X_test_with_missing.index)

        # saving the imputed data, model, and timing information
        save_pickle(imputed_data_path, gain_imputed)

    tf.keras.backend.clear_session()

    return gain_imputed

# -----------------------------------------------------------------------------------------------------------
# Initial or pre-imputation to fill up the original missing values/NaNs for smoother scaling and one-hot encoding processings
# -----------------------------------------------------------------------------------------------------------

# declaring encoder as a global variable
encoder = None
def initial_knn_and_mode_imputed_data(X_train_scaled, X_train_scaled_noisy, X_train_with_missing, X_val_with_missing, X_test_with_missing, fold, config, missing_ratio, repeat):
    """
    Performs initial KNN imputation (numerical) and mode imputation (categorical),
    followed by consistent one-hot encoding across train, validation, and test sets.
    Caches results for efficiency and reproducibility.

    Parameters:
    - X_train_scaled, X_train_scaled_noisy: Original and noise-added scaled training sets.
    - X_train_with_missing, X_val_with_missing, X_test_with_missing: Datasets with induced missing values.
    - fold: Current cross-validation fold.
    - masking_dir: Directory path for storing cached results.
    - missing_ratio: Proportion of induced missingness.
    - repeat: Identifier for experiment repetition.

    Returns:
    Tuple containing:
    - encoder: One-hot encoder fitted on training data.
    - Imputed and one-hot encoded train, validation, and test datasets for further analysis.
    """

    global encoder  # indicating of the global encoder variable
    
    # defining the directories for saving results
    imputation_dir_knn = GET_FINAL_IMPUTATIONS_DIR(config)

    imputer_name = f'KNN_Imputer_for_aes_Fold_{missing_ratio}_fold_{fold}_repeat_{repeat}'
    imputation_file = os.path.join(imputation_dir_knn, f"{imputer_name}.pkl")

    # paths for saving imputed validation and test data
    # file names include fold, missing_ratio and repetition to differentiate between different scenarios
    val_for_aes_imputed_file = os.path.join(imputation_dir_knn, f"X_val_knn_mode_imputed_one_hot_encoded_for_aes_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    test_for_aes_imputed_file = os.path.join(imputation_dir_knn, f"X_test_knn_mode_imputed_one_hot_encoded_for_aes_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    val_for_others_imputed_file = os.path.join(imputation_dir_knn, f"X_val_with_nans_for_others_one_hot_encoded_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    test_for_others_imputed_file = os.path.join(imputation_dir_knn, f"X_test_with_nans_for_others_one_hot_encoded_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    train_noisy_imputed_file = os.path.join(imputation_dir_knn, f"X_train_noisy_one_hot_encoded_for_aes_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    train_for_others_imputed_file = os.path.join(imputation_dir_knn, f"X_train_with_nans_for_others_one_hot_encoded_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    train_imputed_file_for_aes = os.path.join(imputation_dir_knn, f"X_train_one_hot_encoded_for_aes_{missing_ratio}_fold_{fold}_repeat_{repeat}.pkl")
    encoder_file = os.path.join(imputation_dir_knn, f"encoder_fold_{fold}_repeat_{repeat}.pkl")

    # if the imputed data already exists, loading it directly
    if os.path.exists(val_for_aes_imputed_file) and os.path.exists(test_for_aes_imputed_file):
        print(f"Loading initial KNN and mode imputed data for aes for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")
    
        with open(val_for_aes_imputed_file, 'rb') as f:
            X_val_imputed_for_aes_one_hot_encoded = pickle.load(f)
    
        with open(test_for_aes_imputed_file, 'rb') as f:
            X_test_imputed_for_aes_one_hot_encoded = pickle.load(f)

        with open(val_for_others_imputed_file, 'rb') as f:
            X_val_with_missing_one_hot_encoded = pickle.load(f)
    
        with open(test_for_others_imputed_file, 'rb') as f:
            X_test_with_missing_one_hot_encoded = pickle.load(f)
            
        with open(train_for_others_imputed_file, 'rb') as f:
            X_train_with_missing_one_hot_encoded = pickle.load(f)
    
        with open(train_noisy_imputed_file, 'rb') as f:
            X_train_scaled_noisy_one_hot_encoded_for_aes = pickle.load(f)

        with open(train_imputed_file_for_aes, 'rb') as f:
            X_train_scaled_one_hot_encoded_for_aes = pickle.load(f)

        with open(encoder_file, 'rb') as f:
            encoder = pickle.load(f)

        print(f"Imputed data loaded for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}.")
        return (
        encoder,
        X_train_with_missing_one_hot_encoded, 
        X_test_with_missing_one_hot_encoded, 
        X_val_with_missing_one_hot_encoded, 
        X_val_imputed_for_aes_one_hot_encoded, 
        X_test_imputed_for_aes_one_hot_encoded, 
        X_train_scaled_one_hot_encoded_for_aes, 
        X_train_scaled_noisy_one_hot_encoded_for_aes
    )

    # training the KNN imputer
    print(f"Training initial KNN imputer for aes for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")

    knn_imputer_fold = KNNImputer(n_neighbors=5)
    knn_imputer_fold.fit(X_train_scaled.select_dtypes(exclude=['object', 'category']))  # Fit only on numerical data

    # imputing validation and test data
    print(f"Imputing validation and test data for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}...")

    # separate numerical and categorical columns
    numerical_columns = X_train_scaled.select_dtypes(exclude=['object', 'category']).columns
    categorical_columns = X_train_scaled.select_dtypes(include=['object', 'category']).columns

    # KNN Imputation for numerical data (neighbors=5)
    X_val_numerical_imputed = pd.DataFrame(knn_imputer_fold.transform(X_val_with_missing[numerical_columns].values), index=X_val_with_missing.index, columns=numerical_columns)
    X_test_numerical_imputed = pd.DataFrame(knn_imputer_fold.transform(X_test_with_missing[numerical_columns].values), index=X_test_with_missing.index, columns=numerical_columns)


    X_val_categorical_with_nans = X_val_with_missing[categorical_columns].copy()
    X_test_categorical_with_nans = X_test_with_missing[categorical_columns].copy()

    # creating copies to avoid modifying original DataFrames
    X_val_categorical_filled = X_val_with_missing[categorical_columns].copy()
    X_test_categorical_filled = X_test_with_missing[categorical_columns].copy()

    # mode imputation for categorical data
    for col in categorical_columns:
        # mode_value = X_train_scaled[col].mode()[0]  # Get the mode from training set
        mode_value = X_train_scaled[col].dropna().mode()[0]
        X_val_categorical_filled[col] = X_val_categorical_filled[col].fillna(mode_value)  # Fill NaNs with mode in validation set
        X_test_categorical_filled[col] = X_test_categorical_filled[col].fillna(mode_value)  # Fill NaNs with mode in test set

    # combining the imputed numerical and categorical data
    X_val_imputed_for_aes = pd.concat([X_val_numerical_imputed, X_val_categorical_filled], axis=1)
    X_test_imputed_for_aes = pd.concat([X_test_numerical_imputed, X_test_categorical_filled], axis=1)

    # combining the imputed numerical and categorical data
    X_train_imputed_for_others = pd.concat([X_train_with_missing[numerical_columns], X_val_categorical_filled], axis=1)
    X_val_imputed_for_others = pd.concat([X_val_with_missing[numerical_columns], X_val_categorical_filled], axis=1)
    X_test_imputed_for_others = pd.concat([X_test_with_missing[numerical_columns], X_val_categorical_filled], axis=1)

    # applying One-Hot Encoding to All Datasets After Imputation
    print("Applying One-Hot Encoding to Categorical Columns...")

    # fitting the encoder on the training data
    X_train_scaled_one_hot_encoded, encoder = one_hot_encode_for_others(X_train_scaled.copy(), categorical_columns)
    X_train_with_missing_one_hot_encoded, _ = one_hot_encode_for_others(X_train_imputed_for_others.copy(), categorical_columns, encoder)
    X_val_with_missing_one_hot_encoded, _ = one_hot_encode_for_others(X_val_imputed_for_others.copy(), categorical_columns, encoder)
    X_test_with_missing_one_hot_encoded, _ = one_hot_encode_for_others(X_test_imputed_for_others.copy(), categorical_columns, encoder)

    X_train_scaled_one_hot_encoded_for_aes, X_train_scaled_noisy_one_hot_encoded_for_aes, X_val_imputed_for_aes_one_hot_encoded, X_test_imputed_for_aes_one_hot_encoded = consistent_one_hot_encode(X_train_scaled.copy(), X_train_scaled_noisy.copy(), X_val_imputed_for_aes.copy(), X_test_imputed_for_aes.copy(), categorical_columns)

    # **fix: ensuring only test indices are selected**
    test_indices = X_test_with_missing.index  # keeping only test samples
    print(f"Before filtering: X_test_imputed_final shape = {X_test_imputed_for_aes_one_hot_encoded.shape}")
    X_test_imputed_for_aes_one_hot_encoded = X_test_imputed_for_aes_one_hot_encoded.loc[test_indices]
    print(f"After filtering: X_test_imputed_final shape = {X_test_imputed_for_aes_one_hot_encoded.shape}")

    # saving the imputed datasets
    with open(val_for_aes_imputed_file, 'wb') as f:
        pickle.dump(X_val_imputed_for_aes_one_hot_encoded, f)

    with open(test_for_aes_imputed_file, 'wb') as f:
        pickle.dump(X_test_imputed_for_aes_one_hot_encoded, f)

    with open(val_for_others_imputed_file, 'wb') as f:
        pickle.dump(X_val_with_missing_one_hot_encoded, f)

    with open(test_for_others_imputed_file, 'wb') as f:
        pickle.dump(X_test_with_missing_one_hot_encoded, f)
        
    with open(train_for_others_imputed_file, 'wb') as f:
        pickle.dump(X_train_with_missing_one_hot_encoded, f)

    with open(train_imputed_file_for_aes, 'wb') as f:
        pickle.dump(X_train_scaled_one_hot_encoded_for_aes , f)

    with open(train_noisy_imputed_file, 'wb') as f:
        pickle.dump(X_train_scaled_noisy_one_hot_encoded_for_aes , f)

    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder , f)

    print(f"Initial KNN and mode imputed data for aes saved for fold {fold}, missing ratio {missing_ratio}, repetition {repeat}.")

    return (
    encoder,
    X_train_with_missing_one_hot_encoded, 
    X_test_with_missing_one_hot_encoded, 
    X_val_with_missing_one_hot_encoded, 
    X_val_imputed_for_aes_one_hot_encoded, 
    X_test_imputed_for_aes_one_hot_encoded, 
    X_train_scaled_one_hot_encoded_for_aes, 
    X_train_scaled_noisy_one_hot_encoded_for_aes
)