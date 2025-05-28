
# **Thesis: A Comparative Analysis of Advanced Statistical and Machine Learning Methods for Addressing Data Gaps in Tabular Datasets**
_Exploring Deep Learning and Traditional Approaches for Addressing Data Gaps in Tabular Datasets_

---

Welcome to the **main repository** for my **Master's Thesis Project** on **Data Gaps** using both **numerical-only** and **numerical with categorical** datasets. This project compares traditional and deep learning-based imputation techniques by leveraging real-world **Demographic and Health Surveys (DHS)** tabular datasets and synthetic tabular datasets.

---

## **Repository Overview**

This repository is organized into two major subprojects:

1. **Numerical-Only Imputation Pipeline**: Focused on numerical features only.
2. **Numerical with Categorical Imputation Pipeline**: Handles datasets with both numerical and categorical variables.

Each pipeline is independent with its own:
- `main.py`
- `src/`
- `results/`
- `output_images/`

---

## **Project Structure**
_Combining 'Numerical Only' and 'Numerical with Categorical' Data Pipelines_

```text
msc_thesis_2025/
├── README.md                       # Main repository README
├── numerical_only/                 # Numerical-only imputation pipeline
│   ├── README.md                   # README for numerical-only pipeline
│   ├── main.py                     # Main script for numerical-only experiments
│   ├── src/                        # Source code for numerical-only pipeline
│   ├── results/                    # Pickle files for numerical-only results
│   └── output_images/              # Plots & reports for numerical-only
├── numerical_with_categorical/     # Numerical + categorical imputation pipeline
│   ├── README.md                   # README for mixed data pipeline
│   ├── main.py                     # Main script for mixed data experiments
│   ├── src/                        # Source code for mixed data pipeline
│   ├── results/                    # Pickle files for mixed data results
│   └── output_images/              # Plots & reports for mixed data
└── cross-validation/               # Hyperparameter tuning the architectures and other parameetrs for deep learning methods by implementing customized cross-validation
```

Each pipeline follows a similar structure for easy navigation and reproducibility.

---

## **How to Use This Repository**
- Clone this project in a **dedicated local directory** (For example: `~/your_dedicated_local_directory/`).

### Step 1: Clone the repository
```bash
cd ~/your_dedicated_local_directory
git clone https://github.com/kr193/Master-s-Thesis-2025.git
```

### Step 2: Choose your pipeline
For **numerical-only** experiments:
```bash
cd numerical_only
```
For **numerical with categorical** experiments:
```bash
cd numerical_with_categorical
```

### Step 3: Install dependencies
```bash
cd ~/your_dedicated_local_directory/msc_thesis_2025/numerical_only (or numerical_with_categorical)
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4: Run the pipeline
```bash
python main.py
```
You will be guided by **interactive prompts** for only **numerical-only** datasets:

---

## **Subproject READMEs**
- [Numerical-Only Pipeline README](./numerical_only/README.md)
- [Numerical + Categorical Pipeline README](./numerical_with_categorical/README.md)

Each README provides detailed explanations specific to that pipeline, including configurations, evaluation metrics and workflow.

---

## Collaboration & License

- Email: rubaiyakabir11@gmail.com
- License: MIT License

---

## Citation

If using this code for your research, please cite the associated publication.

---

## **Acknowledgments**

Thank you for exploring this project. Any feedback and collaboration are always welcome.

---

