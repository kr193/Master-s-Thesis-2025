#!/usr/bin/env python3
# File: src/gain.py

# ----------------------------------------------------------------------------
# Dataset Type: Aggregated and Numerical DHS Dataset (real-world tabular data)
# ----------------------------------------------------------------------------

# This module defines the complete GAIN pipeline for missing data imputation,
# including normalisation utilities, renormalization, generator and discriminator models,
# custom training loops and inference mechanisms.
#
# Highlights:
# - Custom training logic using Keras and TensorFlow
# - Generator and Discriminator design
# - Batch hint matrix generation for semi-supervised learning
# - Full reset capability for multiple retrainings
# - Supports dynamic missingness patterns.
# - Based on: https://github.com/DeltaFloflo/imputation_comparison
# ---------------------------------------------------------

# -----------------------------------------------------------------------
# GAIN (Generative Adversarial Imputation Networks) - Implementation
# -----------------------------------------------------------------------

# importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# -----------------------
# Normalization Utilities
# -----------------------

def normalization(data, norm_params=None):
    """
    Normalize the data column-wise to a [0, 1] range.
    Args:
        data (numpy array): Input data to normalize.
        norm_params (dict): Optional parameters for normalization (min and max values).

    Returns:
        norm_data (numpy array): Normalized data.
        norm_params (dict): Normalization parameters used.
    """
    N, D = data.shape
    if norm_params is None:
        min_val = np.zeros(D)
        max_val = np.zeros(D)
        norm_data = data.copy()
        for d in range(D):
            m1 = np.nanmin(data[:, d])
            m2 = np.nanmax(data[:, d])
            min_val[d] = m1
            max_val[d] = m2
            norm_data[:, d] = (data[:, d] - m1) / (m2 - m1 + 1e-6)
        norm_params = {"min_val": min_val, "max_val": max_val}
    else:
        min_val = norm_params["min_val"]
        max_val = norm_params["max_val"]
        norm_data = data.copy()
        for d in range(D):
            m1 = min_val[d]
            m2 = max_val[d]
            norm_data[:, d] = (data[:, d] - m1) / (m2 - m1 + 1e-6)
    return norm_data, norm_params

def renormalization(norm_data, norm_params):
    """
    Reverse normalization to original scale.
    Args:
        norm_data (numpy array): Normalized data.
        norm_params (dict): Parameters used for normalization.

    Returns:
        data (numpy array): Data rescaled to original range.
    """
    N, D = norm_data.shape
    min_val = norm_params["min_val"]
    max_val = norm_params["max_val"]
    data = norm_data.copy()
    for d in range(D):
        m1 = min_val[d]
        m2 = max_val[d]
        data[:, d] = norm_data[:, d] * (m2 - m1 + 1e-6) + m1
    return data

# ----------------------
# Model Reset Utility
# ----------------------

def reset_weights(model):
    """
    Completely reinitialize model parameters
    """
    for layer in model.layers:
        if layer.name[:5] == "dense":
            # getting the shape of the kernel (weights) to determine the input and output dimensions
            kernel_shape = layer.kernel.shape  # alternatively, layer.weights[0].shape
            
            nb_in = kernel_shape[0]  # input dimension
            nb_out = kernel_shape[1]  # output dimension
            limit = np.sqrt(6.0 / (nb_in + nb_out))
            
            # reinitializing the kernel (weights) and bias
            r1 = np.random.uniform(-limit, limit, size=kernel_shape)
            r2 = np.zeros(shape=layer.bias.shape)
            
            layer.set_weights([r1, r2])
        
        elif layer.name[:19] == "batch_normalization":
            # getting the shape of the gamma (scaling factor) to initialize batch normalization parameters
            gamma_shape = layer.gamma.shape  # alternatively, layer.weights[0].shape
            
            r1 = np.ones(shape=gamma_shape)  # gamma
            r2 = np.zeros(shape=gamma_shape)  # beta
            r3 = np.zeros(shape=gamma_shape)  # moving mean
            r4 = np.ones(shape=gamma_shape)  # moving variance
            
            layer.set_weights([r1, r2, r3, r4])

# ----------------------------------------
# Mask Distribution and Sampling Functions
# ----------------------------------------

def maskDistribution(dataset):
    """
    unique_masks: list of unique NaN masks found in the dataset
    count_masks: corresponding number of occurrences (the probability distrib.)
    """
    mask = (1.0 - np.isnan(dataset)).astype("int")
    unique_masks = np.unique(mask, axis=0)
    count_masks = np.zeros(len(unique_masks), dtype="int")
    for i1 in range(mask.shape[0]):
        current_mask = mask[i1]
        i2 = np.where((unique_masks == current_mask).all(axis=1))[0][0]
        count_masks[i2] += 1
    return unique_masks, count_masks

def drawMasks(unique_masks, probs, N):
    """
    unique_masks: list of unique masks from which to choose
    probs: vector of probability (should sum up to one)
    N: number of samples to draw
    masks: list of size N containing one mask per row drawn from the desired distribution
    """
    multinom = np.random.multinomial(n=1, pvals=probs, size=N)
    indices = np.where(multinom==1)[1]
    masks = unique_masks[indices]
    return masks

def drawHintMatrix(p, nb_rows, nb_cols):
    """
    Generate a hint matrix for GAIN training.
    Args:
        p (float): Probability of 1s in the hint matrix.
        nb_rows (int): Number of desired rows in the matrix H.
        nb_cols (int): Number of desired columns in the matrix H.

    Returns:
        H (numpy array): Hint matrix.
    """
    H = np.random.uniform(0., 1., size=(nb_rows, nb_cols))
    H = 1.0 * (H < p)
    return H

# --------------------------------------------
# GAIN Generator and Discriminator Builders
# --------------------------------------------

# generator network of GAIN for num+cat dataset
def make_GAINgen(dim):
    """
    Create the generator model for GAIN.
    Args:
        dim (int): Dimensionality of the input data.

    Returns:
        model (tf.keras.Sequential): Generator model.
    """
    model = Sequential()
    model.add(Dense(475, activation="elu", input_shape=(2*dim,)))
    model.add(BatchNormalization())
    model.add(Dense(855, activation="elu"))
    model.add(BatchNormalization())
    model.add(Dense(855, activation="elu"))
    model.add(BatchNormalization())
    model.add(Dense(dim, activation="linear"))
    return model

# discriminator network of GAIN
def make_GAINdisc(dim):
    """
    Create the discriminator model for GAIN.
    Args:
        dim (int): Dimensionality of the input data.

    Returns:
        model (tf.keras.Sequential): Discriminator model.
    """
    model = Sequential()
    model.add(Dense(475, activation="elu", input_shape=(2*dim,))) # specifically manually configured for categorical with numerical dataset
    model.add(Dropout(rate=0.4))
    model.add(Dense(855, activation="elu"))  
    model.add(Dropout(rate=0.4))
    model.add(Dense(855, activation="elu"))  
    model.add(Dropout(rate=0.4))
    model.add(Dense(dim, activation="sigmoid"))
    return model

# -------------------------------------------
# GAIN Code Version 2 (Training + Imputation)
# -------------------------------------------

class GAIN_code_v2:
    """
    GAIN (Generative Adversarial Imputation Networks) implementation for imputing missing data.
    """
    def __init__(self, dim):
        """
        Initialize the GAIN model with generator and discriminator.
        Args:
            dim (int): Dimensionality of the input data.
        """
        self.dim = dim
        self.G = make_GAINgen(dim) # generator
        self.D = make_GAINdisc(dim) # discriminator
        self.Goptim = tf.keras.optimizers.Adam(0.001) # reconstruction vs adversarial loss weight # Weight for reconstruction loss
        self.Doptim = tf.keras.optimizers.Adam(0.001) # % of known features revealed to discriminator
        self.alpha = 50
        self.hint_rate = 0.9
        self.trained = False
        self.nb_epochs = 0
        self.Gloss1 = [] # adversarial loss
        self.Gloss2 = [] # reconstruction loss
        self.Dloss = []

    # discriminator loss: Binary cross-entropy with hint masking
    @staticmethod
    def compute_D_loss(D_output, M, H):
        """
        Compute the discriminator loss during training.
        """
        L1 = M * tf.math.log(D_output + 1e-6)
        L2 = (1.0 - M) * tf.math.log(1.0 - D_output + 1e-6)
        L = - (L1 + L2) * tf.cast((H == 0.5), dtype=tf.float32)
        nb_cells = tf.math.reduce_sum(tf.cast((H == 0.5), dtype=tf.float32))
        return tf.math.reduce_sum(L) / nb_cells if nb_cells > 0 else 0.0

    # generator loss: fool discriminator + minimize reconstruction error
    @staticmethod
    def compute_G_loss(G_output, D_output, X, M, H):
        """
        Compute the generator loss during training.
        """
        Ltemp = - ((1.0 - M) * tf.math.log(D_output + 1e-6))
        L = Ltemp * tf.cast((H == 0.5), dtype=tf.float32)
        nb_cells1 = tf.math.reduce_sum(tf.cast((H == 0.5), dtype=tf.float32))
        loss1 = tf.math.reduce_sum(L) / nb_cells1 if nb_cells1 > 0 else 0.0
        squared_err = ((X - G_output) ** 2) * M
        nb_cells2 = tf.math.reduce_sum(M)
        loss2 = tf.math.reduce_sum(squared_err) / nb_cells2 if nb_cells2 > 0 else 0.0
        return loss1, loss2

    # reset model for fresh training
    def reinitialize(self):
        """
        Reinitialize the weights of both generator and discriminator models.
        
        This is useful for resetting the models to their initial states before retraining.
        Also clears the training history (losses and epoch counter).
        """
        reset_weights(self.G)
        reset_weights(self.D)
        self.trained = False
        self.nb_epochs = 0
        self.Gloss1 = []
        self.Gloss2 = []
        self.Dloss = []

    # single training step (compiled for speed with @tf.function)
    @tf.function  
    def train_step(self, batch_data):
        """
        Perform a single training step for the GAIN model.

        Args:
            batch_data (tf.Tensor): Batch of data with missing values (NaNs).

        Returns:
            G_loss1 (float): Generator adversarial loss for the batch.
            G_loss2 (float): Generator reconstruction loss for the batch.
            D_loss (float): Discriminator loss for the batch.
        
        Steps:
        1. Generate a mask matrix `M` indicating observed values (1) and missing values (0).
        2. Replace missing values with random noise to create `X`.
        3. Train the generator (`G`) and discriminator (`D`) using separate gradient updates.
        4. Calculate and return the losses for monitoring.
        """
        cur_batch_size = batch_data.shape[0]
        noise = tf.random.normal([cur_batch_size, self.dim], dtype=tf.float32)
        batch_data = tf.cast(batch_data, dtype=tf.float32)  # Ensure batch_data is float32
        M = 1.0 - tf.cast(tf.math.is_nan(batch_data), dtype=tf.float32)  # 0=NaN, 1=obs.
        X = tf.where(tf.math.is_nan(batch_data), noise, batch_data)
        G_input = tf.concat((X, M), axis=1)
    
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            G_output = self.G(G_input, training=True)
            X_hat = X * M + G_output * (1.0 - M)
            Htemp = tf.cast(drawHintMatrix(self.hint_rate, cur_batch_size, self.dim), dtype=tf.float32)
            H = M * Htemp + 0.5 * (1.0 - Htemp)
            D_input = tf.concat((X_hat, H), axis=1)
            D_output = self.D(D_input, training=True)
    
            D_loss = self.compute_D_loss(D_output, M, H)
            G_loss1, G_loss2 = self.compute_G_loss(G_output, D_output, X, M, H)
            G_loss = G_loss1 + self.alpha * G_loss2
    
            G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
            D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)
    
            self.Goptim.apply_gradients(zip(G_gradients, self.G.trainable_variables))
            self.Doptim.apply_gradients(zip(D_gradients, self.D.trainable_variables))
    
            return G_loss1, G_loss2, D_loss

    # full training loop
    def train(self, dataset, batch_size, epochs):
        """
        Train the GAIN model on the given dataset.

        Args:
            dataset (numpy array): Dataset containing missing values (NaNs).
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs for training.

        Process:
        - For each epoch, the dataset is divided into batches.
        - Each batch is passed through `train_step` to update the model weights.
        - Losses (adversarial, reconstruction, and discriminator) are recorded for each epoch.
        """
        for epoch in range(epochs):
            G_temp1, G_temp2, D_temp = [], [], []
            for batch_idx in range(0, dataset.shape[0], batch_size):
                batch_data = dataset[batch_idx:batch_idx + batch_size]
                G_loss1, G_loss2, D_loss = self.train_step(batch_data)
                G_temp1.append(G_loss1.numpy())
                G_temp2.append(G_loss2.numpy())
                D_temp.append(D_loss.numpy())
            self.Gloss1.append(np.mean(G_temp1))
            self.Gloss2.append(np.mean(G_temp2))
            self.Dloss.append(np.mean(D_temp))

    # impute missing values using the trained generator
    def impute(self, nandata):
        """
        Impute missing values in the dataset using the trained GAIN model.

        Args:
            nandata (numpy array): Dataset containing missing values (NaNs).

        Returns:
            imputed_data (numpy array): Dataset with missing values replaced by imputed values.
        
        Process:
        - Missing values are replaced by the generator's output.
        - Observed values remain unchanged.
        """
        noise = tf.random.normal([nandata.shape[0], self.dim])
        M_impute = 1.0 - np.isnan(nandata)
        X_impute = tf.where((M_impute == 0.0), noise, nandata)
        G_input = tf.concat((X_impute, M_impute), axis=1)
        G_output = self.G(G_input, training=False)
        imputed_data = (X_impute * M_impute + G_output * (1.0 - M_impute)).numpy()
        return imputed_data