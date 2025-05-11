# **Missing Data Imputation for DHS Indicators**
_Exploring Deep Learning and Traditional Approaches for Data Recovery_

Welcome! This repository contains my Master's research project on **missing data imputation** techniques specifically for **Demographic and Health Surveys (DHS)** datasets. The research compares traditional statistical methods and cutting-edge deep learning models, utilizing both real-world and synthetic data to assess performance.

---

## **Project Overview**

This project:

- ✅ Evaluates various **imputation methods** including **Mean, KNN, MICE**, and deep learning models (**Autoencoder (AE), Denoising AE (DAE), Variational AE (VAE), and GAIN**).
- ✅ Compares their effectiveness using metrics: **RMSE**, **NRMSE**, **R²**, **MAE**, and **Correlation**.
- ✅ Explores missingness types: **MAR (Missing At Random)** and **MCAR (Missing Completely At Random)**.
- ✅ Generates detailed **visualizations** and structured **Excel summaries**.

---

## **Repository Structure**
When cloning this repository, create a dedicated local directory (e.g., final_imputation_project/) to keep things organized. The structure inside your project will look like this:

```bash
msc_thesis_2025/
final_imputation_project/
├── config.json                     # Core configuration file for experiments
├── github_readme.txt                # Draft readme (can be deleted/updated)
├── main.py                          # Main script for running the pipeline
├── src/                             # Source code folder
│   ├── __init__.py
│   ├── config_loader.py
│   ├── cross_val_models_statistics.py
│   ├── deep_learning_methods.py
│   ├── dhs_modelling_functions_new.py
│   ├── evaluation.py
│   ├── features_evaluation.py
│   ├── gain.py
│   ├── helper_functions.py
│   ├── initial_imputation.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   ├── rmse_cv_num_of_features.py
│   ├── run_pipeline.py
│   ├── setup_dirs.py
│   ├── single_imputation.py
│   ├── synthetic_data_generation.py
│   ├── utils.py
│   └── visualization.py
├── data/                            # Contains raw or processed datasets
├── notebooks/                       # Jupyter notebooks for exploration/demos
├── output_images/                   # Contains generated plots and Excel reports
├── results/                         # Auto-generated results (pickles)
│   ├── drop_20_percentage/          # For 20% NaN-drop scenario
│   │   ├── with_masking/            # MAR missingness scenario
│   │   │   └── final_methods_pkls/
│   │   │       ├── task1_final_imputations_missing_ratio/   # Pickle files for Task 1
│   │   │       ├── task2_rmse_vs_num_features/              # Pickle files for Task 2
│   │   │       └── task3_time_vs_missing_ratio/             # Timing metrics (pickles)
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

### Tips:
The project root for any user can be cloned into any local directory ( ~/projects/, ~/research/, etc.).

---

## ⚙️ **Configuration (`config.json`)**

Below is an example configuration (`config.json`) you can use to get started:

```json
{
    "input_dir": "/home/myuser/data/preprocessed_data/DHS_n_more/",
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
    "use_synthetic_data": true,
    "correlation_type": "no_correlation",
    "dim": 96,
    "N": 15000
}
```

You can adjust parameters like `use_synthetic_data`, `correlation_type`, `process_nans`, and the masking option interactively through prompts at runtime or directly via the `config.json` file.

---

## ✅ Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/imputation.git
cd imputation
```

---

## ✅ Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

---

## ✅ Step 3: Run the pipeline

```bash
python main.py
```

You will be guided by **interactive prompts** to configure your experiment settings:

- Choose **Synthetic** or **Real** data.
- For synthetic data, select the **correlation type**:
  - `no_correlation`
  - `medium_correlation`
  - `high_correlation`
- Choose the **missingness type**:
  - `MAR` (Missing At Random)
  - `MCAR` (Missing Completely At Random)

---

## 📊 **Generated Outputs**

After running the experiments, you'll find:

- 📈 **Visualizations**:
  - RMSE & performance metrics across methods.
  - Training and test time comparison.
  - Scatter plots (Actual vs Imputed values).

- 📁 **Excel Reports**:
  - Detailed performance metrics (RMSE, MAE, R², Correlation).

- 📂 **Output Directories**:
  - `results/`
  - `output_images/`

---

## 🔍 **Why is this Important?**

Missing data is a common challenge in large-scale surveys like DHS. This project helps you understand:

- Which **imputation method** works best under various conditions.
- The **trade-offs** between traditional and advanced deep learning methods.
- Provides **practical insights** into handling real-world missing data scenarios.

---

## 🤝 **Want to Collaborate?**

This repository is currently set to private, available only to select collaborators. If you're interested in collaborating or discussing this project, feel free to reach out:

- 📧 **Email**: your.email@example.com  
- 🌐 **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## 📄 **License**

This project is open-sourced under the **MIT License**. Feel free to use, adapt, and distribute as needed!

---

## 🙌 **Acknowledgments**

Thank you for checking out my project! Your feedback and contributions are always welcome. Let's connect!

# 📂 **Project Structure: Numerical + Categorical Dataset**

This project handles **missing data imputation** for datasets containing both **numerical** and **categorical** variables. The methods and evaluation differ from the numerical-only pipeline to accommodate categorical data using techniques like **one-hot encoding**, **precision**, **recall**, **F1-score**, and specialized plots.

---

## 📁 Directory Layout

```
/project_root/
|
|┃-- data/                       # Raw or preprocessed datasets (numerical + categorical)
|┃-- notebooks/                 # Jupyter notebooks (specific to mixed data)
|┃-- results/                   # Auto-generated results (pickle files, metrics)
|┃-- src/                       # Source code for pipeline and utilities
|   |
|   |┃-- __init__.py
|   |┃-- config.py                # Configuration parameters for mixed datasets
|   |┃-- dhs_modelling_functions_new.py
|   |┃-- evaluation.py            # Evaluation: RMSE (numerical), F1, precision, recall (categorical)
|   |┃-- gain.py                  # GAIN imputation model
|   |┃-- imputations.py           # Imputation methods for both numerical & categorical
|   |┃-- masking.py               # Masking strategies for MAR/MCAR scenarios
|   |┃-- preprocessing.py         # One-hot encoding, label encoding, scaling
|   |┃-- requirements.txt         # Python dependencies
|   |┃-- run_pipeline.py          # Main orchestration for pipeline execution
|   |┃-- utils.py                 # Helper functions
|   └-- visualization.py         # Plots: RMSE, precision, recall, F1-score
|┃-- main.py                    # Main script to run the pipeline (with interactive prompts)
└-- README.md                   # Project overview & instructions
```

---

## 📊 **Key Differences (Compared to Numerical-Only Project)**

- **Data Handling**:
  - Includes **categorical columns** (e.g., type of residence, water source).
  - **One-hot encoding** for multi-class categorical columns.
  - **Label encoding** for binary categorical columns.

- **Imputation Methods**:
  - Support both **numerical** and **categorical** columns.
  - Categorical columns are imputed using modified deep learning models (e.g., GAIN) with **sigmoid activations**.

- **Evaluation Metrics**:
  - **Numerical** columns: RMSE, MAE, R², correlation.
  - **Categorical** columns: Precision, Recall, F1-score (column-wise and row-wise).

- **Visualizations**:
  - **RMSE vs Missing Ratio** for numerical columns.
  - **Precision, Recall, F1-score plots** for categorical columns.
  - **Scatter plots** for actual vs imputed values (numerical).

---

## 🔄 Workflow Overview

1. **Preprocessing**:
   - One-hot encode categorical columns.
   - Scale numerical columns.

2. **Imputation**:
   - Apply methods like **Mean**, **KNN**, **MICE**, **GAIN**, etc.
   - For categorical columns, use sigmoid activations in neural nets.

3. **Evaluation**:
   - RMSE, MAE, R², Correlation for numerical.
   - Precision, Recall, F1-score for categorical.

4. **Visualization**:
   - Generate separate plots for numerical and categorical metrics.

---

## 📚 Example Evaluation Metrics

- **Numerical**:
  - RMSE: 0.245 (Mean ± Std)
  - R²: 0.92

- **Categorical**:
  - Precision: 0.88
  - Recall: 0.85
  - F1-score: 0.86

---

## 📢 Notes for Users

- Clone this project in a **dedicated local directory** (e.g., `~/projects/missing_data_imputation/`).
- The **`main.py`** script will guide you through setting **synthetic vs real data**, **correlation types**, **missingness types (MAR/MCAR)** interactively.
- Results and plots are saved automatically under **results/** and **output_images/**.

---

## 👥 Collaboration & License

- 📧 Email: your.email@example.com
- 👤 LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/yourprofile)
- 📜 License: MIT License

---

Thank you for exploring this project! Your feedback and collaboration are always welcome.

# 📚 **Missing Data Imputation for DHS Indicators - Master Repository**
_Combining Numerical-Only and Numerical + Categorical Data Pipelines_

Welcome to the **master repository** for my **Master's research project** on **missing data imputation** using both **numerical-only** and **numerical + categorical** datasets. This project compares traditional and deep learning-based imputation techniques, leveraging real-world DHS data and synthetic datasets.

---

## 📍 **Repository Overview**

This repository is organized into two major subprojects:

1. **Numerical-Only Imputation Pipeline**: Focused on numerical features only.
2. **Numerical + Categorical Imputation Pipeline**: Handles datasets with both numerical and categorical variables.

Each pipeline is independent with its own:
- `main.py`
- `src/`
- `results/`
- `output_images/`

---

## 🔍 **Project Structure**

```bash
msc_thesis_2025/
├── README.md                       # Master repository README
├── numerical_only/                 # Numerical-only imputation pipeline
│   ├── README.md                   # README for numerical-only pipeline
│   ├── main.py                     # Main script for numerical-only experiments
│   ├── src/                        # Source code for numerical-only pipeline
│   ├── results/                    # Pickle files for numerical-only results
│   └── output_images/              # Plots & reports for numerical-only
├── numerical_categorical/          # Numerical + categorical imputation pipeline
│   ├── README.md                   # README for numerical + categorical pipeline
│   ├── main.py                     # Main script for mixed data experiments
│   ├── src/                        # Source code for mixed data pipeline
│   ├── results/                    # Pickle files for mixed data results
│   └── output_images/              # Plots & reports for mixed data
└── data/                           # Shared datasets (real or synthetic)
```

Each pipeline follows a similar structure for easy navigation and reproducibility.

---

## 💡 **How to Use This Repository**

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/imputation.git
cd imputation
```

### Step 2: Choose your pipeline
- For **numerical-only** experiments:
  ```bash
  cd numerical_only
  ```
- For **numerical + categorical** experiments:
  ```bash
  cd numerical_categorical
  ```

### Step 3: Install dependencies
```bash
pip install -r src/requirements.txt
```

### Step 4: Run the pipeline
```bash
python main.py
```
You will be guided by **interactive prompts**:
- Choose **Synthetic** or **Real** data.
- Select **correlation type** (for synthetic data).
- Choose **missingness type** (MAR/MCAR).

---

## 🔹 **Subproject READMEs**
- [Numerical-Only Pipeline README](./numerical_only/README.md)
- [Numerical + Categorical Pipeline README](./numerical_categorical/README.md)

Each README provides detailed explanations specific to that pipeline, including configurations, evaluation metrics, and workflow.

---

## 💼 **License**
This project is licensed under the **MIT License**. Feel free to use, adapt, and distribute.

---

## 👥 **Contact & Collaboration**

If you're interested in collaborating or discussing this project:
- 📧 **Email**: your.email@example.com
- 🌐 **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

Thank you for exploring this project! Check individual READMEs for more details.

