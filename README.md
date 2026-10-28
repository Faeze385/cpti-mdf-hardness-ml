# cpti-mdf-hardness-ml

Machine learning prediction of Vickers hardness in multi-directionally forged (MDF), intermediate-annealed commercially pure titanium (CP-Ti), comparing SVR, Random Forest, KNN, and gradient boosting.

## Overview

This project predicts Vickers hardness from the processing state of CP-Ti grade 2 that was multi-directionally forged with intermediate annealing at 650 °C. The dataset comes from the experimental study by Zare & Mostafaei (2025) (see [Source study](#source-study)).

Each sample is labelled with a code describing its ordered sequence of forging (**F**) and annealing (**A**) steps, for example `F`, `FA`, `FAF`, `FAFAFF`, or `As-received`. Instead of one-hot encoding this label, the project turns it into order-aware numerical features. This matters physically: a sequence that ends on annealing is recrystallized and relatively soft, while a sequence that ends on forging keeps a high dislocation density and is harder.

Four regression models are trained, tuned, and compared:

- Support Vector Regression (SVR)
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- Gradient Boosting (`HistGradientBoostingRegressor`, used as a stand-in for XGBoost)

## Source study

The data originates from:

> Zare, M., & Mostafaei, M.A. (2025). Intermediate annealing of severely deformed pure titanium by multi-directional forging: Effect on mechanical properties and microstructure. *Materials Science & Engineering A*, 927, 147998. https://doi.org/10.1016/j.msea.2025.147998

Please cite this paper if you use the data or build on this work.

## Dataset

81 hardness measurements across the processing conditions As-received, F, FF, FA, FAF, FAFA, FAFAF, FAFF, and FAFAFF.

| Column | Meaning |
|---|---|
| `Sample` | F/A processing-sequence code |
| `Distance` | Indentation position across the sample cross-section (mm) |
| `Hardness` | Vickers microhardness, the **target** |
| `n-pass` | Number of forging cycles (count of F) |
| `n-Anneal` | Number of anneals (count of A) |
| `n-A/n-F` | Ratio of anneals to forging cycles |
| `ơ-flow` | Flow stress |
| `Ꜫ-acc` | Accumulated strain |

> **Note:** the interpretation of `ơ-flow` and `Ꜫ-acc` is inferred from how the values line up with the source paper (UTS as flow stress; 0.47 strain per forging cycle). Verify against your original data before relying on it.

> **Data availability:** the dataset is derived from Zare & Mostafaei (2025), whose data is available from the authors on request. It is **not redistributed** in this repository. To run the notebook, place your own copy as `Book1.xlsx` in the project root with the columns listed above.

## Feature engineering

`n-pass`, `n-Anneal`, and `n-A/n-F` are just counts of F and A in the `Sample` code and say nothing about order. So the `Sample` string is parsed into twelve order-aware features:

| Feature | Description |
|---|---|
| `seq_length` | Total number of steps (0 for As-received) |
| `first_is_F`, `last_is_F` | Whether the sequence starts / ends on a forging step |
| `frac_F` | Proportion of steps that are forging |
| `n_transitions`, `norm_transitions` | Number / normalized number of F↔A alternations |
| `first_A_pos`, `last_A_pos`, `first_F_pos`, `last_F_pos` | Normalized (0–1) position of the first / last occurrence of each step type |
| `max_F_run`, `max_A_run` | Longest run of consecutive identical steps |
| `ends_with_anneal` | Whether the final step is an anneal |

The final 15-feature input is these twelve plus `Distance`, `ơ-flow`, and `Ꜫ-acc`. The three count columns are dropped because the sequence features already contain that information.

## Method

- 80/20 train/test split (`random_state=42`)
- Hyperparameters tuned with `GridSearchCV` (5-fold CV, scored on R²)
- `StandardScaler` for SVR and KNN
- Evaluation: test R², RMSE, MAE, plus 5-fold cross-validated R² on the full dataset
- Feature importance: native impurity-based importance for Random Forest, permutation importance (drop in test R²) for the other three models

## Results

| Model | Test R² | Test RMSE | Test MAE | 5-fold CV R² |
|---|---|---|---|---|
| SVR | 0.936 | 11.23 | 10.34 | 0.961 ± 0.015 |
| Random Forest | 0.944 | 10.53 | 9.54 | 0.969 ± 0.016 |
| KNN | 0.927 | 12.04 | 10.31 | 0.959 ± 0.020 |
| **Gradient Boosting** | **0.955** | **9.46** | **8.41** | **0.969 ± 0.011** |

Gradient boosting performs best, with Random Forest close behind. Across models, flow stress is the strongest predictor, followed by sequence features such as `frac_F`, `last_is_F` / `ends_with_anneal`, and `max_F_run`.

With only 81 measurements across 9 processing conditions, treat these numbers as indicative rather than definitive. The model should not be extrapolated to conditions outside those sampled, such as the 450 °C annealing branch of the source study.

## Repository structure

```
cpti-mdf-hardness-ml/
├── hardness_prediction.ipynb   # full pipeline: features, tuning, evaluation, plots
├── requirements.txt
├── .gitignore                  # excludes the dataset (Book1.xlsx)
└── README.md
```

`Book1.xlsx` (the dataset) is intentionally not included. See [Dataset](#dataset).

## Getting started

```bash
git clone https://github.com/<Faeze385>/cpti-mdf-hardness-ml.git
cd cpti-mdf-hardness-ml
pip install -r requirements.txt
# add your copy of the dataset as Book1.xlsx in this folder, then:
jupyter notebook hardness_prediction.ipynb
```

`requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
openpyxl
jupyter
```

The notebook reads `Book1.xlsx` from the same folder and will fail without it.

## Notes

- `xgboost` was not available in the environment where this was developed, so `HistGradientBoostingRegressor` (scikit-learn's histogram-based gradient-boosted trees) is used instead. To use XGBoost, install it and replace the model with `xgboost.XGBRegressor`, adjusting the parameter grid accordingly.
- Results depend on the random split. Cross-validated scores are reported alongside the single test split for that reason.

## Related work

The approach draws on earlier machine-learning studies of hardness prediction:

- Shozib, I.A. et al. (2021). Modelling and optimization of microhardness of electroless Ni–P–TiO₂ composite coating based on machine learning approaches and RSM. *J. Mater. Res. Technol.*, 12, 1010–1025.
- Jeon, J. et al. (2021). Application of Machine Learning Algorithms and SHAP for Prediction and Feature Analysis of Tempered Martensite Hardness in Low-Alloy Steels. *Metals*, 11(8), 1159.
- Swetlana, S., Khatavkar, N., & Singh, A.K. (2020). Development of Vickers hardness prediction models via microstructural analysis and machine learning. *J. Mater. Sci.*, 55, 15845–15856.
- Geng, X. et al. (2022). Prediction of Hardenability Curves for Non-Boron Steels via a Combined Machine Learning Model. *Materials*, 15(9), 3127.

## License

Code license: add one of your choice (for example MIT). The dataset is not part of this repository and remains subject to the terms of the original source study.

