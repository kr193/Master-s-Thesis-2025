📘 README — Missing Data Imputation Project

This project performs advanced imputation on numerical datasets (like DHS) using deep learning and classical methods.

👣 Getting Started:
-----------------------
1. Clone or download this repository.
2. Put the DHS dataset (CSV format) inside the `/data/` folder. Name it: `dataset.csv`
3. Make sure all modules from `/src/` are placed correctly.
4. Install required Python libraries (requirements.txt to be added).
5. Run the pipeline with:
    python main.py

📂 Folder Structure:
-----------------------
.
├── data/
│   └── dataset.csv
├── results/
│   ├── with_masking/
│   │   └── final_methods_pkls/
│   │       └── task1_final_imputations_missing_ratio/
│   ├── without_masking/
        └── final_methods_pkls/
            └── task1_final_imputations_missing_ratio/
│   
├── src/
│   └── All utility and model code files
├── config.py
└── main.py

📌 Main Features:
-----------------------
- Works with numerical household survey data.
- Modular functions in `src/` directory.
- Deep models: AE, DAE, VAE, GAIN.
- Metrics: RMSE, MAE, R², Correlation, Time.
- Visuals: RMSE plots, scatter plots, time vs missingness.

---

🛠 Maintained by: Rubaiya
📅 Last Updated: April 2025
