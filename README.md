# Heart Disease Prediction

A binary classifier that predicts whether a patient has heart disease, based on clinical measurements such as age, cholesterol, chest pain type and exercise ECG results.

Built from scratch as a learning project to practice the full machine learning workflow: data exploration, preprocessing, model comparison and honest evaluation.

## Data

[UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) ([Kaggle mirror](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)): 920 patients collected at four clinics (Cleveland, Hungary, Switzerland, VA Long Beach).

The original target `num` (0–4) describes disease severity. It was converted to a binary target: `0` = healthy, `1–4` = sick (about 55 % of patients are sick).

## Approach

- **Dropped columns:** `id`, `thal` and `ca` (more than 50 % missing), and `dataset` (the clinic, see *Findings*)
- **Invalid values:** a cholesterol value of 0 is physically impossible, so these were treated as missing
- **Missing values:** median imputation for numeric features, an explicit `Unknown` category for categorical features
- **Preprocessing:** `StandardScaler` for numeric features, `OneHotEncoder` for categorical features, combined in a `ColumnTransformer`
- **Pipeline:** preprocessing and model wrapped in one scikit-learn `Pipeline`, so every cross-validation fold is preprocessed using only its own training data (no data leakage)
- **Split:** stratified 75/25 train/test split; the test set was only used once for the final model
- **Model selection:** 5-fold cross-validation on the training set, comparing Logistic Regression, Random Forest and XGBoost

## Results

5-fold cross-validation on the training set (mean ± standard deviation):

| Model | Accuracy | Recall |
|---|---|---|
| Baseline (always predicts "sick") | 0.552 | – |
| **Logistic Regression** | **0.817 ± 0.028** | **0.853 ± 0.026** |
| Random Forest | 0.809 ± 0.013 | 0.843 ± 0.022 |
| XGBoost | 0.786 ± 0.042 | 0.827 ± 0.039 |

Final evaluation of Logistic Regression on the held-out test set: **accuracy 0.817, recall 0.843**, consistent with the cross-validation estimate.

## Findings

**The clinic was a shortcut.** At first, the strongest feature in the model was which clinic a patient came from. I removed it, without any loss in accuracy, so the model learns from medical values only and is more likely to work for patients from other clinics.

**Logistic Regression and Random Forest performed equally well.** The difference between them is smaller than the variation across cross-validation folds. I chose Logistic Regression because it is simpler and easier to understand.

**Recall matters more than accuracy here.** Accuracy is the share of all predictions that are correct. Recall is the share of sick patients that the model actually recognises as sick. Recall is more important for this task, because a sick patient classified as healthy is worse than a healthy patient classified as sick.

## Limitations

- Small dataset (920 patients) collected in the 1980s
- Missing values are not random: some clinics never recorded certain tests. The `Unknown` categories (for example `fbs_Unknown`) carry large weights, so they may still partly act as a proxy for the clinic.
- Not a medical tool: the model has not been validated on any external patient data

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

This Readme was enhanced with AI