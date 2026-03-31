# 📱 Telecom Customer Targeting Using Machine Learning

A machine learning project built to help a fictional telecom company (Wallace Communications) 
stop wasting money on cold calls by predicting which customers are actually 
likely to sign up for a new mobile contract.

---

## 🎯 The Problem

Wallace Communications has spent years building out their landline network 
across UK cities. Now they want to break into the mobile market — but running 
a call centre campaign for 50,000+ customers is expensive and not viable.

The challenge: **who do you call?**

This project builds predictive models to identify customers most likely to 
take out a new contract, so the marketing team can focus their efforts where 
it actually counts.

---

## 📊 Dataset

- 50,662 customer records from previous marketing campaigns
- Mix of demographic, financial, and campaign history features
- Target variable: `new_contract_this_campaign` (yes / no)
- Class imbalance: ~80% no, ~20% yes

---

## What I Actually Did

### Data Quality
Before touching a model, I audited every column for inconsistencies.
Found two encoding errors caught through bar plot inspection:

- `has_tv_package` contained `'n'` (5 records) — abbreviation of `'no'`,
  standardised
- `last_contact` contained `'cell'` (2 records) — duplicate of `'cellular'`,
  merged

Small numbers, but the kind of silent errors that corrupt categorical
encodings downstream if left unchecked.

### Feature Engineering
Dropped non-predictive identifiers (`ID`, `town`, `country`) and a
low-signal date field. Made deliberate decisions about numeric vs categorical
treatment — for example, `conn_tr` was retained as numeric (ordinal grouping)
rather than one-hot encoded, with the trade-off documented: treating an
arbitrary group ID as ordinal imposes an ordering that may not reflect
reality, potentially missing non-linear group effects.

### Preprocessing
Built a `ColumnTransformer` pipeline:
- **Numeric features** → `StandardScaler` (essential for LR and MLP
  sensitivity to feature scale)
- **Categorical features** → `OneHotEncoder` with `drop='first'` to avoid
  multicollinearity

Critically — the scaler was **fit only on training data** and applied to
validation and test sets separately. Fitting on the full dataset leaks
distributional information into evaluation and is one of the most common
methodological mistakes in ML pipelines.

### Train / Validate / Test Split
**60 / 20 / 20** with stratified sampling across all three sets.

The validate set was used exclusively for hyperparameter selection. The test
set was not touched until a final model was chosen — this is the correct
methodology and the only way to get an honest generalisation estimate.

### Modelling & Hyperparameter Tuning

All three model families were tuned via grid search evaluated on validation
ROC-AUC:

| Model | Hyperparameters Tuned | Best Val ROC-AUC |
|---|---|---|
| Logistic Regression | C ∈ {0.001, 0.01, 0.1, 1.0, 10.0} | 0.764 |
| Decision Tree | max_depth ∈ {3,5,7,10}, min_samples_leaf ∈ {20,50,100} | 0.774 |
| Neural Network (MLP) | architecture × alpha | 0.786 |

**Why ROC-AUC?**  
With an 80/20 class split, accuracy is a trap. A model predicting "no" for
every customer scores 80% accuracy while being completely useless. ROC-AUC
measures discriminatory power across all decision thresholds — it directly
captures whether the model can rank likely converters above non-converters,
which is exactly what the business needs.

`class_weight='balanced'` was applied across all models to prevent the
majority class from dominating the loss function.

---

## Key Findings

### Model Comparison
All three models were tuned and evaluated on a properly held-out test set.
The Neural Network came out on top across every metric that matters:

| Model | ROC-AUC | F1-Macro | Accuracy |
|---|---|---|---|
| Neural Network ✅ | 0.788 | 0.704 | 84.1% |
| Decision Tree | 0.774 | 0.676 | 76.6% |
| Logistic Regression | 0.764 | 0.665 | 75.1% |

### What the Confusion Matrix Actually Tells Us
On the 10,133 customer test set the Neural Network produced:

| | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | 7,718 ✅ | 435 ❌ |
| **Actual: Yes** | 1,171 ❌ | 809 ✅ |

In plain business terms:
- **809 genuine converters** were correctly identified and targeted —
  these are the calls that actually result in a signed contract
- **435 customers** were incorrectly flagged — wasted calls, but a
  manageable number given the volume
- **1,171 actual converters** were missed — customers who would have
  signed up but weren't targeted

### The Real Business Impact
Without any model, calling all 10,133 customers gives a **19.6% hit rate**
— roughly 1 in 5 calls converts.

With the model targeting only the **1,244 flagged customers**, the hit rate
jumps to **65%** — nearly 2 in 3 calls converts.

That means Wallace Communications needs **less than half the calls** to
reach the same pool of willing customers. Scaled across 50,000+ customers,
the call-centre cost saving is substantial.

### Strongest Predictor — Previous Campaign Outcome
Previous campaign outcome was by far the most powerful feature in the
dataset. Customers with a prior **successful** outcome converted at over
**3× the overall average rate**. Customers with a prior **failure** outcome
converted *below* the baseline — meaning chasing already-resistant customers
without a new offer is a net cost, not an opportunity.

Actionable segmentation this produces:
- **Priority tier:** customers with a prior success — highest expected value per call
- **Standard tier:** never-contacted customers — near-baseline conversion,
  large volume, worth targeting at scale
- **Deprioritise:** customers with a prior failure — below-baseline conversion,
  not worth the spend

### Data Quality Issues Caught
Two silent encoding errors were found during exploratory analysis that would
have corrupted categorical encodings downstream if left unchecked:
- `has_tv_package`: `'n'` → `'no'` (5 records)
- `last_contact`: `'cell'` → `'cellular'` (2 records)

---

## 📓 View the Full Analysis

Click **[Telecom Customer Targeting.ipynb](./Telecom%20Customer%20Targeting.ipynb)**
to view the complete notebook with all plots, model results, and commentary
rendered inline — no setup required.

---

## Project Structure
```
├── Telecom Customer Targeting.ipynb   # Full reproducible pipeline
├── wallacecommunications.csv          # Source dataset
└── README.md
```

---

## Reproducing the Results
```bash
git clone https://github.com/Imran3285/telecom-customer-targeting-using-ML.git
cd telecom-customer-targeting-using-ML
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook "Telecom Customer Targeting.ipynb"
```

Run all cells top to bottom. Random seed is fixed at 42 — results are
fully reproducible.

---

## Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

---

## Author

**Muhammad Imran**  
Data Scientist 
GitHub: [@Imran3285](https://github.com/Imran3285)  
Email: m.imran88222@gmail.com
