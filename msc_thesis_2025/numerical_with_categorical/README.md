# Categorical Imputation Evaluation

This repository contains code for evaluating imputation techniques on datasets with both numerical and categorical variables under varying missing data ratios.

## Structure

- `main.py` - Main entry script to control evaluation
- `load_data.py` - Handles data loading and preprocessing
- `masking.py` - Introduces missingness under MAR/MCAR
- `imputation.py` - Runs imputation methods like AE, DAE, VAE, GAIN
- `categorical_metrics.py` - Computes evaluation metrics (RMSE, F1, etc.)
- `visualization.py` - Generates performance plots

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage

You can run `main.py` to start the full imputation pipeline.

## Citation

If using this code for your research, please cite the associated publication.
