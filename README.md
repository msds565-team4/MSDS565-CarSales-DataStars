
# MSDS565 — Car Sales Price Prediction | Team DataStars

**Course:** MSDS 565 — Predictive Modeling & Analytics  
**Project:** Unit 1 — Tabular Regression (Car Sales)

## Team Members

Listed alphabetically by last name:

- Antonio Brown
- Michael Flanders
- Justin Irick
- India Scruggs

## Project Overview

This project applies the CRISP-DM workflow to predict used-car listing prices using vehicle characteristics and listing information.

The project consists of five sequential Jupyter notebooks covering:

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Regression Modeling
4. Feature Selection
5. Model Interpretability and Fairness Analysis

The target variable is **`price`**.

## Dataset

The dataset is the instructor-provided, modified US Used Cars dataset, distributed through the course Google Drive folder.

**Dataset filename:** `CarSales_Large.csv`

**Dataset dimensions:** 302,590 rows and 66 columns.

The public Kaggle version is for background reference only. The instructor-provided version should be used because it contains course-specific modifications.

Raw and intermediate CSV files are stored locally rather than committed to this repository.

Notebook file paths may require adjustment depending on the user's local directory structure.

## Repository Structure

```text
MSDS565-CarSales-DataStars/
│
├── README.md
├── environment.yml
│
├── models/
│   ├── P1_N3_linreg.pkl
│   ├── P1_N3_ridge.pkl
│   ├── P1_N3_nn2.keras
│   ├── P1_N4_linreg_selected.pkl
│   ├── P1_N4_ridge_selected.pkl
│   └── P1_N4_nn2_selected.keras
│
└── notebooks/
    ├── P1_CarSales-1_EDA_DataStars.ipynb
    ├── P1_CarSales-2_Preprocessing_DataStars.ipynb
    ├── P1_CarSales-3_Modeling_DataStars.ipynb
    ├── P1_CarSales-4_FeatureSelection_DataStars.ipynb
    └── P1_CarSales-5_xAI_DataStars.ipynb
```

The two large Random Forest models are distributed separately through GitHub Releases.

## Notebook Workflow

### Notebook 1 — Exploratory Data Analysis

**File:** `P1_CarSales-1_EDA_DataStars.ipynb`

This notebook examines:

- Dataset dimensions and structure
- Missing values and data types
- Feature distributions
- Outliers
- Relationships between vehicle characteristics and price
- Data-quality concerns

The findings guide the preprocessing decisions in Notebook 2.

### Notebook 2 — Data Preprocessing

**File:** `P1_CarSales-2_Preprocessing_DataStars.ipynb`

This notebook prepares the dataset for modeling through:

- Missing-value handling
- Data-type corrections
- Outlier treatment
- Feature engineering
- Categorical encoding
- Feature scaling
- Training and testing data preparation

The dataset is divided into training and testing sets using an 80/20 split with `random_state=42`.

**Primary outputs:**

- `processed_train.csv`
- `processed_test.csv`

These files are saved locally.

### Notebook 3 — Regression Modeling

**File:** `P1_CarSales-3_Modeling_DataStars.ipynb`

This notebook evaluates multiple regression approaches:

- Linear Regression
- Random Forest Regression
- Ridge Regression
- Neural Networks using Keras

Five-fold cross-validation and hyperparameter tuning are used to compare model performance.

**Evaluation metrics:**

- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)
- Model training time

**Primary outputs:**

- `model_metrics.csv`
- Trained regression models
- Selected neural network model

### Notebook 4 — Feature Selection

**File:** `P1_CarSales-4_FeatureSelection_DataStars.ipynb`

This notebook evaluates whether reducing the number of predictors improves model efficiency while maintaining predictive performance.

Feature selection includes:

- Random Forest feature importance
- Selection of the top 100 features
- Variance Inflation Factor (VIF) analysis
- Removal of highly collinear predictors
- Model retraining and performance comparisons

The notebook reports reducing the original 293 predictors to 94 selected predictors.

**Primary outputs:**

- `selected_train.csv`
- `selected_test.csv`
- `model_metrics_selected.csv`
- Feature-selected regression models

### Notebook 5 — Model Interpretability and Fairness

**File:** `P1_CarSales-5_xAI_DataStars.ipynb`

This notebook uses explainable AI and subgroup analysis to investigate model predictions.

**Interpretability methods:**

- SHAP feature importance
- SHAP beeswarm plots
- SHAP force plots
- Analysis of accurate and inaccurate predictions

**Subgroup analysis:**

- Vehicle make (`make_name`)
- Body type (`body_type`)
- Fuel type (`fuel_type`)

Fairlearn is used to compare model performance across vehicle subgroups.

These analyses examine predictive performance disparities across vehicle categories rather than demographic fairness.

## Model Performance

Notebook 3 reports the following held-out test results:

| Model | Test MAE | Test MAPE | Test R² |
|---|---:|---:|---:|
| Linear Regression | Approximately $5,498 | 26.86% | See Notebook 3 |
| Random Forest | $1,638.97 | 6.92% | 0.9503 |
| Ridge Regression | $5,494.76 | 26.84% | 0.7340 |

The Random Forest model demonstrated the strongest performance among the three traditional regression models shown above.

Additional neural network results, training times, and cross-validation comparisons are documented in Notebook 3.

### Feature-Selected Model Evaluation

Notebook 4 evaluates model performance after reducing the feature set.

Notebook 5 separately documents a rebuilt-model evaluation with the following results:

| Metric | Result |
|---|---:|
| MAE | $1,666.53 |
| MAPE | 7.18% |
| R² | 0.9577 |
| Test observations | 59,243 |

These Notebook 5 results represent a rebuilt-model run and should not be interpreted as an exact reproduction of Notebook 4's saved-model evaluation.

## Saved Models

The project includes trained models generated in Notebooks 3 and 4.

### Models Stored in the GitHub Repository

The following smaller models belong in the repository's `models/` directory:

| Model File | Notebook |
|---|---|
| `P1_N3_linreg.pkl` | Notebook 3 |
| `P1_N3_ridge.pkl` | Notebook 3 |
| `P1_N3_nn2.keras` | Notebook 3 |
| `P1_N4_linreg_selected.pkl` | Notebook 4 |
| `P1_N4_ridge_selected.pkl` | Notebook 4 |
| `P1_N4_nn2_selected.keras` | Notebook 4 |

**Repository update — November 17, 2026:**

The team confirms that `P1_N4_ridge_selected.pkl` has been added to the GitHub repository.

### Large Random Forest Models — GitHub Releases

Two previously trained Random Forest models exceed GitHub's standard individual file size limit for Git repositories.

| Model File | Notebook | Approximate Size |
|---|---|---|
| `P1_N3_rf.pkl` | Notebook 3 | 3.61 GB |
| `P1_N4_rf_selected.pkl` | Notebook 4 | 3.63 GB |

To preserve these trained models without rerunning computationally expensive training procedures, the files were archived using 7-Zip.

**Archive settings:**

- Format: 7z
- Compression: Store (no compression)
- Split volume size: 1500M
- Distribution: GitHub Releases

### Published GitHub Release

**Release:** Team DataStars — Trained Random Forest Models

**Version:** `v1.0-models`

**Download link:**

https://github.com/msds565-team4/MSDS565-CarSales-DataStars/releases/tag/v1.0-models

The team confirms that all six archive parts have been successfully uploaded and are visible under the published release's **Assets** section.

The release contains three numbered archive parts for each Random Forest model.

### Instructions for Restoring Random Forest Models

1. Open the GitHub Release link above.
2. Download all three numbered archive parts for the desired model.
3. Place the downloaded archive parts together in the same local folder.
4. Open the `.7z.001` file using 7-Zip.
5. Extract the original `.pkl` model file.
6. Place the extracted model in the repository's `models/` directory.

The reconstructed files should be named:

```text
models/P1_N3_rf.pkl
models/P1_N4_rf_selected.pkl
```

All archive parts must be present for successful extraction.

**Note:** The archive assets are hosted in GitHub Releases rather than directly tracked in the repository. Download integrity and successful reconstruction have not been independently verified.

Only load serialized pickle models from trusted sources and use a compatible Python and scikit-learn environment.

## Environment Setup

The project uses Python and the packages specified in `environment.yml`.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/msds565-team4/MSDS565-CarSales-DataStars.git
cd MSDS565-CarSales-DataStars
```

### Step 2 — Create the Conda Environment

```bash
conda env create -f environment.yml
```

### Step 3 — Activate the Environment

Check the `name:` field in `environment.yml`, then activate that environment.

```bash
conda activate YOUR_ENVIRONMENT_NAME
```

### Step 4 — Obtain the Dataset

Download the instructor-provided `CarSales_Large.csv` from the course Google Drive.

Place it in the local data directory used by the notebooks.

Update file paths where necessary.

### Step 5 — Restore Saved Models

Download and extract the two Random Forest models from GitHub Releases if they are needed for inference or further analysis.

### Step 6 — Run Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in numerical order:

```text
Notebook 1 → Notebook 2 → Notebook 3
           → Notebook 4 → Notebook 5
```

Each notebook should be configured to locate its required input files and save its outputs to the appropriate directories.

## Reproducibility Notes

The repository contains notebooks with saved execution outputs documenting previous runs.

The following considerations apply:

- The original dataset is distributed through the course rather than GitHub.
- Raw and intermediate CSV files are stored locally.
- Local paths may need modification.
- Large Random Forest models must be downloaded from GitHub Releases.
- Neural network results may vary depending on random initialization and package versions.
- Notebook 5 documents a rebuilt-model evaluation.
- A complete fresh execution from a clean environment has not been independently verified.

The GitHub Release approach preserves access to the trained Random Forest models while avoiding the need to rerun expensive training procedures.

## Model Interpretation and Limitations

SHAP analysis helps explain how individual features contribute to model predictions.

These explanations describe model behavior and should not be interpreted as evidence of causal relationships.

Subgroup comparisons may reflect differences in vehicle prices, sample sizes, feature distributions, and model performance.

Results should therefore be interpreted within the context of the available data.

## Project Status

As of November 17, 2026:

- All five project notebooks have been prepared.
- The model development and feature-selection analyses have been completed.
- The team confirms that the Ridge model `P1_N4_ridge_selected.pkl` has been added to GitHub.
- Both large Random Forest models are available as multipart archives through the published GitHub Release.
- All six numbered archive parts have been confirmed visible by the team.
- Repository documentation includes model restoration and reproducibility instructions.

**Team DataStars — MSDS565 Predictive Modeling & Analytics**

