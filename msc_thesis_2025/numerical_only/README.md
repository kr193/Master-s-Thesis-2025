# **Sub-Project Structure: Only Numerical Datasets: Addressing Data Gaps in Tabular Datasets**
*Exploring Deep Learning and Traditional Approaches for Addressing Data Gaps in Numerical Data Only*

This folder contains code for evaluating imputation techniques on datasets containing only **numerical features** which are aggregated data, extensively preprocessed from DHS program under varying missing data ratios.

---

## **Sub-Project Overview**

This project:
-  Works with numerical household survey data from DHS program and synthetic datasets.
-  Applies **one-hot encoding** for multi-class categorical columns and **scaling** for numerical columns.
-  Evaluates various **imputation methods** including **Mean, KNN, MICE-Ridge**, and deep learning methods (**Autoencoder (AE), Denoising AE (DAE), Variational AE (VAE) and GAIN**).
-  Compares their effectiveness using metrics: **RMSE**, **R²**, **MAE** and **Correlation** across **five folds** over **six missing ratios** via customized cross-validation.
-  Explores missingness types: **MAR (Missing At Random)** and **MCAR (Missing Completely At Random)**.
-  Generates detailed **visualizations** and structured **Excel summaries**.

---

## **Sub-Project Structure**
Clone this project in a **dedicated local directory** (for example: `~/your_dedicated_local_directory/`).


```text
numerical_only/
├── __init__.py                      # Marks numerical_only folder as a package
├── config.json                      # Core configuration file for experiments
├── main.py                          # Main script for running the 'numerical_only' pipeline
├── requirements.txt                 # Python dependencies
├── src/                             # Source code for entire imputation and evaluation
│   ├── __init__.py                  # Marks src folder as a package
│   ├── config_loader.py             # Loads and parses config.json
│   ├── deep_learning_methods.py     # AE, DAE, VAE architecture and training OF AE, DAE, VAE, GAIN
│   ├── dhs_modelling_functions_new.py # DHS-specific preprocessing utilities
│   ├── evaluation.py                # Computes RMSE, MAE, R² and Correlation
│   ├── gain.py                      # GAIN imputation implementation
│   ├── helper_functions.py          # Misc. helpers: reproducibility, seeding, directory checks
│   ├── initial_imputation.py        # Applies initial KNN imputation on missing spaces to fill NaNs
│   ├── load_data.py                 # Loads datasets
│   ├── preprocessing.py             # Scaling and normalization of numerical data
│   ├── run_pipeline.py              # Runs complete pipeline for given config and user choices
│   ├── setup_dirs.py                # Creates result and output folders based on parameters
│   ├── single_imputation.py         # Applies Mean, KNN, MICE-Ridge
│   ├── synthetic_data_generation.py # Generates synthetic data with different correlation levels
│   ├── utils.py                     # General-purpose utilities and constants
│   └── visualization.py             # Plots: Mean (RMSE, MAE, time and correlation) visuals
├── data/                            # Contains extensively processed dataset as pickle file
├── notebooks/                       # Jupyter notebooks for running entire pipeline
├── output_images/                   # Contains generated plots and Excel reports
├── results/                         # Auto-generated results (pickles)
│   ├── drop_20_percentage/          # For 20% NaN-drop scenario
│   │   ├── with_masking/            # MAR missingness scenario
│   │   │   └── final_methods_pkls/
│   │   │       ├── task1_final_imputations_missing_ratio/   # Pickle files for Task 1
│   │   │       ├── task2_rmse_vs_num_features/              # Pickle files for Task 2
│   │   │       └── task3_time_vs_missing_ratio/             # Pickle files for Task 3
│   │   └── without_masking/         # MCAR missingness scenario
│   │       └── final_methods_pkls/
│   │           ├── task1_final_imputations_missing_ratio/
│   │           ├── task2_rmse_vs_num_features/
│   │           └── task3_time_vs_missing_ratio/
│   ├── drop_all_nans/
│   │   ├── with_masking/
│   │   └── without_masking/
│   │       └── final_methods_pkls/
│   │           └── (Same as above)
│   └── keep_all_numerical/
│       ├── with_masking/
│       └── without_masking/
│           └── final_methods_pkls/
│               └── (Same as above)
└── synthetic_data/                  # Synthetic data results
    ├── synthetic_high_correlation/  # High correlation scenario
    │   ├── with_masking/
    │   │   └── final_methods_pkls/
    │   │       ├── task1_final_imputations_missing_ratio/
    │   │       ├── task2_rmse_vs_num_features/
    │   │       └── task3_time_vs_missing_ratio/
    │   └── without_masking/
    │       └── final_methods_pkls/
    │           └── (Same as above)
    ├── synthetic_medium_correlation/
    │   └── (Same structure as above)
    └── synthetic_no_correlation/
        └── (Same structure as above)
```
---

## **Configuration (config.json)**

Below is an example configuration (config.json) you can use to get started:

```json
{
    "input_dir": "data",
    "dataset_type": "HR",
    "group_by_col": "adm2_gaul",
    "urban_rural_all_mode": "all",
    "drop_agriculture": false,
    "scale_numerical_data": true,
    "drop_countries": true,
    "egypt_dropping": true,
    "drop_columns": [
        "Meta; adm0_gaul",
        "Meta; GEID_init"
    ],
    "countries_to_drop": [
        "Egypt",
        "Comoros",
        "Central African Republic"
    ],
    "missingness_fraction": 0.3,
    "use_synthetic_data": false,
    "correlation_type": "no_correlation",
    "dim": 96,
    "N": 15000
}
```

You can adjust parameters like **use_synthetic_data**, **egypt_dropping** etc. directly via the config.json file and also select the **masking** and **process_nans** options interactively through prompts at runtime. 

---

##  Step 1: Clone the repository

```bash
cd ~/your_dedicated_local_directory
git clone https://github.com/kr193/Master-s-Thesis-2025.git
```

---

##  Step 2: Install dependencies

```bash
cd ~/your_dedicated_local_directory/msc_thesis_2025/numerical_only
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

---

##  Step 3: Run the pipeline

```bash
python main.py
```
Or run the notebook `final_numerical_dataset.ipynb` inside the `notebooks` folder.

You will be guided by **interactive prompts** to configure your experiment settings:

- Choose **Synthetic** or **Real** data.
- For real data, select the **process_nans** option:
  - keep_all_numerical
  - drop_all_nans
  - drop_20_percentage
- For synthetic data = true from config.json, select the **correlation type**:
  - no_correlation
  - medium_correlation
  - high_correlation
- Choose the **missingness type**:
  - MAR (Missing At Random)
  - MCAR (Missing Completely At Random)

---

## **Generated Outputs**

After running the experiments, you will find:

- **Visualizations**:
  - Mean RMSE plots across missing ratios.
  - Mean Training and test time comparison across missing ratios.
  - Mean RMSE across number of features.
  - Boxplots for Mean RMSE Std Dev across methods over 10%-60% missing ratios.
  - Scatter plots for specific fold (Actual vs Imputed values).

- **Excel Reports**:
  - Detailed performance metrics (Mean RMSE, Mean MAE, Mean R², Mean Correlation over five folds).

- **Output Directories**:
  - results/
  - output_images/

---

## **Why is this Important?**

Data Gaps are a common challenge in large-scale surveys like DHS. This project helps you understand:

- Which **imputation method** works best under various conditions.
- The **trade-offs** between traditional and advanced deep learning methods.
- Provides **practical insights** into handling real-world data gaps or missing data scenarios.

---
## **Collaboration & License**

* Email: [rubaiyakabir11@gmail.com](mailto:rubaiyakabir11@gmail.com)
* License: MIT License

---

## **Citation**

If using this code for research, please cite the corresponding publication.

---

## **Acknowledgments**

Thank you for exploring this sub-project. Feedback and collaboration are warmly welcomed.

---