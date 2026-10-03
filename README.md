# Corporate Bankruptcy Prediction

UE24CS352A Machine Learning, Mini-Project

Predicting whether a company will go bankrupt within the next few years from 64 financial ratios, using the UCI **Polish Companies Bankruptcy** dataset. Only about 5% of firms go bankrupt, so the project focuses on handling class imbalance and on metrics that measure how many bankruptcies are actually caught.

## Results at a glance

Main experiment: 3rd-year file (outcome 3 years ahead), 10,416 firms after removing duplicates, 4.7% bankrupt. Stratified 80/20 split; models compared by 5-fold stratified cross-validation on the training set; the test set is used once at the end.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Baseline (always "safe") | 0.047 | 0.048 | 0.500 | 0.00 | 0.00 | 0.00 |
| Naive Bayes | 0.111 | 0.111 | 0.743 | 0.42 | 0.10 | 0.17 |
| Logistic Regression | 0.233 | 0.256 | 0.830 | 0.33 | 0.27 | 0.30 |
| SVM (RBF) | 0.292 | 0.330 | 0.858 | 0.44 | 0.31 | 0.37 |
| Random Forest | 0.356 | 0.453 | 0.918 | 0.51 | 0.42 | 0.46 |
| Neural Network (MLP) | 0.451 | 0.535 | 0.863 | 0.50 | 0.62 | 0.55 |
| LightGBM | 0.624 | 0.725 | 0.964 | 0.68 | 0.68 | 0.68 |
| **XGBoost (final model)** | **0.636** | **0.737** | **0.966** | **0.58** | **0.77** | **0.66** |

Recall, precision and F1 use each model's decision threshold tuned on cross-validation predictions. PR-AUC of a random guess is 0.05.

Key findings:
- **Accuracy is misleading here.** A model that calls every firm "safe" is 95% accurate and catches zero bankruptcies.
- **Gradient boosting wins clearly.** XGBoost and LightGBM reach about 3x the PR-AUC of logistic regression and over 2x that of the SVM, because tree ensembles cope with the extreme outliers, nonlinear ratio effects and missing values in this data.
- **The decision threshold matters as much as resampling.** With threshold tuning, plain XGBoost matched class-weighted XGBoost; SMOTE oversampling did worse (`results/imbalance_study.csv`).
- **The model works at every horizon**, from 5 years to 1 year before the outcome (ROC-AUC 0.95 to 0.97; best 1 year ahead).
- **Most important ratio:** Attr27, operating profit / financial expenses, i.e. whether a firm earns enough to cover its interest costs.

## Repository layout

```
├── app.py                  Streamlit demo app
├── data/                   UCI dataset, one ARFF file per forecasting year (1year … 5year)
├── src/
│   ├── config.py           paths, random seed, descriptions of the 64 ratios
│   ├── data.py             loading, duplicate removal, stratified split
│   ├── models.py           all models as leakage-free pipelines
│   ├── evaluate.py         metrics, cross-validation, threshold tuning
│   ├── plots.py            chart styling
│   ├── train.py            runs every experiment, saves results, figures and the model
│   └── predict.py          scores firms with the saved model (command line)
├── models/                 trained model (created by train.py)
└── results/                CSV tables and figures (created by train.py)
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone <your-repo-url>
cd bankruptcy-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

1. **Train and evaluate everything** (about 5 to 6 minutes on a laptop):
   ```bash
   python src/train.py
   ```
   Use `python src/train.py --quick` (about 3 minutes) to skip the SVM and the 5-year horizon study.
   Outputs: `results/*.csv`, `results/figures/*.png`, `models/xgboost_model.json` + `models/model_info.json`.

2. **Predict from the command line:**
   ```bash
   python src/predict.py                         # 10 firms from the test set, with their real outcome
   python src/predict.py --csv firms.csv --out scored.csv   # your own file with columns Attr1..Attr64
   ```

3. **Interactive demo:**
   ```bash
   streamlit run app.py
   ```
   Pick a firm from the held-out test set (or upload a CSV) to see its bankruptcy probability, its most influential ratios, and the full results.

## Method

1. **Data cleaning.** 87 exact duplicate rows are removed before splitting, so no firm appears in both train and test.
2. **Missing values.** 43% of rows have at least one missing ratio. Median imputation (robust to outliers) for models that need it; XGBoost and LightGBM handle missing values natively.
3. **Scaling.** Ratios are heavy-tailed (Attr5 ranges from -11.9 million to 685,000). A quantile transform to a normal shape is used for logistic regression, SVM, Naive Bayes and the MLP. Tree models need no scaling.
4. **Class imbalance.** Class weights (logistic regression, SVM, random forest, `scale_pos_weight` for boosting) or SMOTE inside the pipeline (Naive Bayes, MLP, which have no class-weight option).
5. **Leakage-free evaluation.** All preprocessing sits inside scikit-learn / imbalanced-learn pipelines, so imputation, scaling and SMOTE are fit only on training folds.
6. **Model selection** by mean 5-fold CV PR-AUC. XGBoost had the highest CV score and was chosen before looking at the test set.
7. **Decision threshold** chosen to maximise F1 on the out-of-fold CV predictions, never on the test set.
8. **Interpretation** with permutation importance on the test set (drop in PR-AUC when a feature is shuffled).

## Dataset

Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). *Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction.* Expert Systems with Applications, 58, 93–101. Data: UCI Machine Learning Repository, "Polish Companies Bankruptcy".

Each file `kyear.arff` holds financial ratios from year *k* of the observation period and the bankruptcy status after (6 − k) years. Bankrupt firms were observed 2000–2012, operating firms 2007–2013.
