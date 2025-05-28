# **Numerical with Categorical Datasets: Addressing Data Gaps in Mixed-Type Tabular Datasets**

*Exploring Deep Learning and Traditional Approaches for Addressing Missing Data in Numerical + Categorical Dataset*

This folder contains code for evaluating imputation techniques on datasets containing both **numerical and categorical features** (sourced from DHS program), under varying missing data ratios.

---

## **Sub-Project Overview**

This project:

* Works with **mixed-type household survey data** (numerical + categorical) from DHS program.
* Applies **one-hot encoding** for multi-class categorical columns and **scaling** for numerical columns.
* Evaluates **imputation methods** including **Mean, KNN, MICE-Ridge** and deep learning methods (**Autoencoder (AE), Denoising AE (DAE), Variational AE (VAE), and GAIN**).
* Measures:
  * **Numerical metrics**: RMSE, MAE, R², Correlation.
  * **Categorical metrics**: Accuracy, Precision, Recall, and F1-score.
* Evaluates results across **five folds** over **six missing ratios**.
* Supports **MAR (Missing At Random)** and **MCAR (Missing Completely At Random)** scenarios.
* Generates detailed **visualizations** and structured **Excel summaries**.

---

## **Sub-Project Structure**

Clone this project in a **dedicated local directory** (for example: `~/your_dedicated_local_directory/`).

```text
numerical_with_categorical/
├── __init__.py                      # Marks mixed_data folder as a package
├── main.py                          # Main script for running the 'mixed_data' pipeline
├── requirements.txt                 # Python dependencies (root-level)
├── src/                             # Source code for entire imputation and evaluation
│   ├── __init__.py                  # Marks src as a package
│   ├── config.py                    # Configuration parameters for mixed datasets
    ├── load_data.py                 # Loads datasets
│   ├── dhs_modelling_functions_new.py # DHS-specific preprocessing utilities
│   ├── evaluation.py                # Evaluation: RMSE (numerical), F1, precision, recall (categorical)
│   ├── gain.py                      # GAIN imputation model
│   ├── imputations.py               # Imputation methods for both numerical & categorical
│   ├── masking.py                   # Masking strategies for MAR/MCAR scenarios
│   ├── preprocessing.py             # One-hot encoding, and scaling
│   ├── run_pipeline.py              # Main orchestration for pipeline execution
│   ├── utils.py                     # Helper functions
    ├── setup_dirs.py                # Creates result and output folders based on parameters
│   └── visualization.py             # Plots: Mean RMSE and Mean Accuracy plots
├── data/                            # preprocessed datasets (numerical + categorical)
├── notebooks/                       # Jupyter notebooks (specific to mixed data)
├── results/                         # Auto-generated results (pickle files, metrics)
└── README.md                        # Project overview & instructions
```
---

## **Run Instructions**

### Step 1: Clone the repository

```bash
cd ~/your_dedicated_local_directory
git clone https://github.com/kr193/Master-s-Thesis-2025.git
```

### Step 2: Install dependencies

```bash
cd ~/your_dedicated_local_directory/msc_thesis_2025/numerical_with_categorical
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the pipeline

```bash
python main.py
```

Or run the notebook `final_notebook_with_categoricals.ipynb` inside the `notebooks` folder.

---

## **Generated Outputs**

* **Visualizations**:

  * Mean RMSE plots for numerical columns
  * Mean Accuracy plots for categorical columns

* **Excel Reports**:

  * RMSE, MAE, R², Correlation (numerical)
  * Accuracy, Precision, Recall, F1-score (categorical)

* **Output Directories**:

  * results

---

## **Why is this Important?**

Handling both numerical and categorical missing data is crucial for real-world surveys like DHS. This project enables:

* A comprehensive comparison of **imputation strategies** across data types
* Understanding performance trade-offs of **deep learning vs traditional methods**
* Practical guidance for **mixed-type imputation tasks** at scale

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
