#!/usr/bin/env python3
# File: src/gain.py

# ----------------------------------------------------------------------------
# Dataset Type: Numerical with Categorical (mixed-type real-world DHS dataset)
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

# importing required libraries
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# -----------------------
# Normalization Utilities
# -----------------------

# required TensorFlow Keras components for the GAIN model
# normalizes numerical data between [0, 1] for stable training
def normalization(data, norm_params=None, numeric_mask=None):
    """
    Normalizes numerical columns in the dataset to [0, 1] range.
    """
    N, D = data.shape
    norm_data = data.copy()

    if numeric_mask is None:
        numeric_mask = np.array([np.issubdtype(data[:, d].dtype, np.number) for d in range(D)])

    if norm_params is None:
        # computing min and max for each column
        min_val = np.zeros(D)
        max_val = np.zeros(D)
        for d in range(D):
            if numeric_mask[d]:
                m1 = np.nanmin(data[:, d])
                m2 = np.nanmax(data[:, d])
                min_val[d] = m1
                max_val[d] = m2
                # applying min-max normalization
                norm_data[:, d] = (data[:, d] - m1) / (m2 - m1 + 1e-6)
        norm_params = {"min_val": min_val, "max_val": max_val}
    else:
        # uses previously stored parameters
        min_val = norm_params["min_val"]
        max_val = norm_params["max_val"]
        for d in range(D):
            if numeric_mask[d]:
                m1 = min_val[d]
                m2 = max_val[d]
                norm_data[:, d] = (data[:, d] - m1) / (m2 - m1 + 1e-6)

    return norm_data, norm_params

# reverts min-max normalization
def renormalization(norm_data, norm_params):
    """
    Reverses normalization back to original scale.
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

# reinitializes all model weights (for clean retraining)
def reset_weights(model):
    """
    Reinitialize weights of all layers in a Keras model.
    """
    """Completely reinitialize model parameters"""
    for layer in model.layers:
        if layer.name[:5] == "dense":
            # getting the shape of the kernel (weights) to determine the input and output dimensions
            # Xavier/Glorot uniform initialization
            kernel_shape = layer.kernel.shape  # alternatively, layer.weights[0].shape
            
            nb_in = kernel_shape[0]  # input dimension
            nb_out = kernel_shape[1]  # output dimension
            limit = np.sqrt(6.0 / (nb_in + nb_out))
            
            # reinitialize the kernel (weights) and bias
            r1 = np.random.uniform(-limit, limit, size=kernel_shape)
            r2 = np.zeros(shape=layer.bias.shape)
            
            layer.set_weights([r1, r2])
        
        elif layer.name[:19] == "batch_normalization":
            # getting the shape of the gamma (scaling factor) to initialize batch normalization parameters
            # resetting batch norm parameters
            gamma_shape = layer.gamma.shape  # alternatively, layer.weights[0].shape
            
            r1 = np.ones(shape=gamma_shape)  # gamma
            r2 = np.zeros(shape=gamma_shape)  # beta
            r3 = np.zeros(shape=gamma_shape)  # moving mean
            r4 = np.ones(shape=gamma_shape)  # moving variance
            
            layer.set_weights([r1, r2, r3, r4])

# ----------------------------------------
# Mask Distribution and Sampling Functions
# ----------------------------------------

# extracts distribution of missingness masks in dataset
def maskDistribution(dataset):
    """
    Get the unique missingness patterns and their counts from dataset.
    """
    """unique_masks: list of unique NaN masks found in the dataset
    count_masks: corresponding number of occurrences (the probability distrib.)"""
    mask = (1.0 - np.isnan(dataset)).astype("int")
    unique_masks = np.unique(mask, axis=0)
    count_masks = np.zeros(len(unique_masks), dtype="int")
    for i1 in range(mask.shape[0]):
        current_mask = mask[i1]
        i2 = np.where((unique_masks == current_mask).all(axis=1))[0][0]
        count_masks[i2] += 1
    return unique_masks, count_masks

# samples new missingness patterns from the mask distribution
def drawMasks(unique_masks, probs, N):
    """
    Sample N new masks from the known distribution.
    """
    """unique_masks: list of unique masks from which to choose
    probs: vector of probability (should sum up to one)
    N: number of samples to draw
    masks: list of size N containing one mask per row drawn from the desired distribution"""
    multinom = np.random.multinomial(n=1, pvals=probs, size=N)
    indices = np.where(multinom==1)[1]
    masks = unique_masks[indices]
    return masks

# creates the hint matrix that reveals part of the mask to the discriminator
def drawHintMatrix(p, nb_rows, nb_cols):
    """
    Generate hint matrix with given probability.
    """
    """p: probability of ones
    nb_rows: number of desired rows in the hint matrix H
    nb_cols: number of desired columns in the hint matrix H
    H: hint matrix"""
    H = np.random.uniform(0., 1., size=(nb_rows, nb_cols))
    H = 1.0 * (H < p)
    return H

# --------------------------------------------
# GAIN Generator and Discriminator Builders
# --------------------------------------------

# generator network of GAIN
def make_GAINgen(dim):
    """
    Generator network: predicts missing values.
    """
    model = Sequential([
        Dense(512, activation="relu", kernel_regularizer=l2(1e-4), input_shape=(2*dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(dim, activation="sigmoid")
    ])
    return model

# discriminator network of GAIN
def make_GAINdisc(dim):
    """
    Discriminator network: distinguishes observed vs. imputed entries.
    """
    model = Sequential([
        Dense(512, activation="relu", input_shape=(2*dim,), kernel_regularizer=l2(1e-4)), # specifically manually configured for categorical with numerical dataset
        Dropout(0.3),
        Dense(512, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(dim, activation="sigmoid")
    ])
    return model

# -------------------------------------------
# GAIN Code Version 2 (Training + Imputation)
# -------------------------------------------

# the GAIN_code_v2 class
class GAIN_code_v2:
    def __init__(self, dim, cat_mask=None):
        self.dim = dim
        self.G = make_GAINgen(dim) # generator
        self.D = make_GAINdisc(dim) # discriminator
        self.Goptim = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.Doptim = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.alpha = 100 # reconstruction vs adversarial loss weight # Weight for reconstruction loss
        self.hint_rate = 0.9 # % of known features revealed to discriminator
        self.trained = False
        self.nb_epochs = 0
        self.Gloss1 = [] # adversarial loss
        self.Gloss2 = [] # reconstruction loss
        self.Dloss = []
        self.cat_mask = tf.convert_to_tensor(cat_mask, dtype=tf.float32) if cat_mask is not None else None

    # discriminator loss: Binary cross-entropy with hint masking
    @staticmethod
    def compute_D_loss(D_output, M, H):
        """
        Discriminator loss (binary cross-entropy)
        """
        L1 = M * tf.math.log(D_output + 1e-6)
        L2 = (1.0 - M) * tf.math.log(1.0 - D_output + 1e-6)
        L = - (L1 + L2) * tf.cast((H == 0.5), dtype=tf.float32)
        nb_cells = tf.math.reduce_sum(tf.cast((H == 0.5), dtype=tf.float32))
        return tf.math.reduce_sum(L) / nb_cells if nb_cells > 0 else 0.0
        
    # generator loss: fool discriminator + minimize reconstruction error
    def compute_G_loss(self, G_output, D_output, X, M, H):
        """
        Generator loss: adversarial + reconstruction components
        """
        Ltemp = - ((1.0 - M) * tf.math.log(D_output + 1e-6))
        L = Ltemp * tf.cast((H == 0.5), dtype=tf.float32)
        nb_cells1 = tf.math.reduce_sum(tf.cast((H == 0.5), dtype=tf.float32))
        loss1 = tf.math.reduce_sum(L) / nb_cells1 if nb_cells1 > 0 else 0.0

        squared_err = ((X - G_output) ** 2) * M
        if self.cat_mask is not None:
            weighted_err = squared_err * (2.0 * self.cat_mask + 1.0 * (1.0 - self.cat_mask))
        else:
            weighted_err = squared_err

        nb_cells2 = tf.math.reduce_sum(M)
        loss2 = tf.math.reduce_sum(weighted_err) / nb_cells2 if nb_cells2 > 0 else 0.0
        return loss1, loss2
        
    # reset model for fresh training
    def reinitialize(self):
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
        One training iteration for both generator and discriminator.
        """
        cur_batch_size = batch_data.shape[0]
        noise = tf.random.normal([cur_batch_size, self.dim], dtype=tf.float32)
        batch_data = tf.cast(batch_data, dtype=tf.float32)
        M = 1.0 - tf.cast(tf.math.is_nan(batch_data), dtype=tf.float32)
        X = tf.where(tf.math.is_nan(batch_data), noise, batch_data)
        G_input = tf.concat((X, M), axis=1)

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            G_output = self.G(G_input, training=True)
            X_hat = X * M + G_output * (1.0 - M)
            # constructs hint matrix
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
    def train(self, dataset, batch_size=8192, epochs=1):
        """
        Train GAIN model on the full dataset.
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
        Impute missing entries in the input using trained generator.
        """
        noise = tf.random.normal([nandata.shape[0], self.dim])
        M_impute = 1.0 - np.isnan(nandata)
        X_impute = tf.where((M_impute == 0.0), noise, nandata)
        G_input = tf.concat((X_impute, M_impute), axis=1)
        G_output = self.G(G_input, training=False)
        imputed_data = (X_impute * M_impute + G_output * (1.0 - M_impute)).numpy()
        return imputed_data