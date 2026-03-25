# 📱 Telecom Customer Targeting Using Machine Learning

A machine learning project built as part of my postgraduate studies.  
The goal is simple — help a fictional telecom company (Wallace Communications) 
stop wasting money on cold calls by predicting which customers are actually 
likely to sign up for a new mobile contract.

---

## 🧠 The Problem

Wallace Communications has spent years building out their landline network 
across UK cities. Now they want to break into the mobile market — but running 
a call centre campaign for 50,000+ customers is expensive.

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

## ⚙️ Models Built

| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |
| Neural Network (MLP) | scikit-learn |

All models were tuned using a **60/20/20 train/validate/test split** with 
stratified sampling to preserve class balance across all sets.

---

## 📈 Evaluation Metric

Given the class imbalance, **ROC-AUC** was used as the primary metric rather 
than accuracy — a model that predicts "no" for everyone would score 80% 
accuracy while being completely useless in practice.

---

## 🔍 Key Findings

- The **Neural Network** achieved the best ROC-AUC on the test set (~0.788)
- **Previous campaign outcome** was the strongest predictor of conversion —  
  customers who responded positively before are far more likely to convert again
- Two data quality issues were found and fixed during cleaning:
  - `has_tv_package`: value `'n'` was a typo for `'no'` (5 records)
  - `last_contact`: value `'cell'` was a duplicate of `'cellular'` (2 records)

---

## 🗂️ Project Structure
```
├── wallace_ml_assignment.ipynb   # Full analysis notebook
├── wallacecommunications.csv     # Dataset
├── error_detection_barplots.png  # Data quality check plots
├── confusion_matrix_final.png    # Final model confusion matrix
├── roc_curves.png                # ROC curves for all models
├── feature_importance.png        # Decision tree feature importances
└── README.md
```

---

## 🚀 How to Run

1. Clone the repo
```bash
   git clone https://github.com/Imran3285/Telecom-Customer-Targeting-Using-ML.git
```

2. Install dependencies
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Launch the notebook
```bash
   jupyter notebook wallace_ml_assignment.ipynb
```

---

## 🛠️ Tools & Libraries

- Python 3
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---

## 👤 Author

**Muhammad Imran**  
Junior Data Scientist  
GitHub: [@Imran3285](https://github.com/Imran3285)

---

*This project is part of a university assignment. The dataset and scenario are fictional.*
