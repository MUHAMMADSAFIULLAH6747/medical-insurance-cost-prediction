# Medical Insurance Cost Prediction — Full Project Documentation

## 1. Problem Statement

Medical insurers need consistent, data-driven estimates of expected healthcare charges for new policyholders. Manual underwriting is slow and can vary by reviewer. This project builds a supervised regression model that predicts annual medical insurance charges (USD) from demographic, lifestyle, and health attributes.

**Business value**
- Faster premium estimation for underwriting workflows
- Transparent risk drivers (especially smoking status)
- Reproducible, auditable prediction pipeline

**ML formulation**
- Task: Supervised learning — regression
- Target: `charges` (continuous USD)
- Primary model: Multiple Linear Regression

## 2. Dataset

| Item | Detail |
|---|---|
| Source | Kaggle: `mirichoi0218/insurance` (Medical Cost Personal Dataset) |
| Format | CSV |
| Raw size | 1,338 rows × 7 columns |
| Clean size | 1,337 rows × 9 numeric columns (after encoding) |
| Missing values | 0 |
| Duplicates removed | 1 |

### Features

| Variable | Role | Type | Description |
|---|---|---|---|
| `charges` | Target | Numeric | Annual medical insurance charges (USD) |
| `age` | Feature | Integer | Beneficiary age (18–64) |
| `sex` | Feature | Categorical | `female` / `male` |
| `bmi` | Feature | Numeric | Body Mass Index (kg/m²) |
| `children` | Feature | Integer | Number of dependents (0–5) |
| `smoker` | Feature | Categorical | `yes` / `no` |
| `region` | Feature | Categorical | `northeast`, `northwest`, `southeast`, `southwest` |

### Key exploratory findings
- Mean age ≈ 39.2 years; mean BMI ≈ 30.7 (overweight/obese range)
- Charges are right-skewed (mean ≈ $13,270; median ≈ $9,382; max ≈ $63,770)
- Non-smokers ≈ 79.5%; smokers ≈ 20.5%
- Smokers show substantially higher charge distributions in EDA plots

## 3. Data Preprocessing

1. **Missing values:** none found → no imputation
2. **Duplicates:** one duplicate removed → 1,337 rows
3. **Encoding**
   - `sex`: female=0, male=1
   - `smoker`: no=0, yes=1
   - `region`: one-hot encoding with `drop_first=True`
     → `region_northwest`, `region_southeast`, `region_southwest`
     (northeast is the baseline)

Final ML matrix: 8 predictors + target `charges`.

## 4. Exploratory Analysis Summary

Important patterns used for modeling decisions:
1. Target distribution is skewed with high-cost outliers.
2. Smoking status creates a clear charge separation (boxplot).
3. BMI vs charges interaction is stronger among smokers.
4. Age generally increases charges; smoker hue shows parallel bands.
5. Correlation structure supports linear modeling, with `smoker` as the dominant associate of `charges`.

## 5. Selected Model & Training Process

**Model:** `sklearn.linear_model.LinearRegression`

**Why Linear Regression**
- Interpretable coefficients for business explanation
- Appropriate baseline for continuous charge prediction
- Fast to train and deploy

**Training protocol**
- Feature matrix `X`: all columns except `charges`
- Target `y`: `charges`
- Split: 80% train / 20% test (`random_state=42`)
  - Train: 1,069 rows
  - Test: 268 rows
- Fit ordinary least squares on training data
- Evaluate on held-out test set

Reproduce training:

```bash
python src/download_data.py
python src/train_model.py
```

## 6. Evaluation Results

Results from the reproducible training script (aligned with notebook Phase 5/6 outputs):

| Metric | Value |
|---|---|
| Training R² | ≈ 0.7299 (72.99%) |
| Testing R² | ≈ 0.8069 (80.69%) |
| MAE | ≈ $4,177.05 |
| RMSE | ≈ $5,956.34 |

### Coefficient interpretation (approximate)

| Feature | Coefficient | Meaning |
|---|---|---|
| Intercept | ≈ -$11,092.65 | Baseline offset |
| age | ≈ +$248.21 | Cost rises with age |
| sex (male) | ≈ -$101.54 | Small gender effect |
| bmi | ≈ +$318.70 | Higher BMI → higher cost |
| children | ≈ +$533.01 | More dependents → higher cost |
| smoker | ≈ +$23,077.76 | Dominant risk driver |
| region_* | negative vs NE | Mild regional adjustments |

**Finding:** Smoking status dominates predicted cost. Holding other inputs fixed, being a smoker increases predicted charges by roughly $23k.

## 7. Prediction Application

Interactive Streamlit app (`app.py`) collects:
- Age, BMI, Children, Gender, Smoker, Region

Then encodes inputs exactly like training and returns estimated USD charges.

```bash
streamlit run app.py
```

### Example test cases (trained model)

| Case | Age | BMI | Children | Smoker | Region | Predicted cost |
|---|---|---|---|---|---|---|
| Low risk | 30 | 25.0 | 0 | no | southeast | **$3,380.74** |
| High risk | 45 | 32.0 | 2 | yes | northwest | **$33,925.76** |
| App demo | 30 | 25.0 | 0 | no | northeast | **$4,219.66** |

## 8. Limitations

1. Linear model may underfit non-linear smoker–BMI interactions.
2. Dataset is relatively small (~1.3k rows) and U.S.-centric.
3. No temporal / medical-history features (claims history, chronic conditions).
4. Errors can be large for extreme high-cost cases (RMSE ≈ $6k).
5. Predictions are estimates for education/demo use — not actuarial quotes.

## 9. Reproducibility Checklist

- [x] Dataset download script
- [x] Shared preprocessing module
- [x] Training script with fixed random seed
- [x] Saved model artifact (`models/insurance_cost_model.joblib`)
- [x] Saved metrics JSON
- [x] Evaluation figure export
- [x] Streamlit deployment app
- [x] README with setup + results
