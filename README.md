# 📡 Customer Churn Prediction & Analysis

> A full-stack data science project predicting telecom customer churn using Machine Learning, SQL, and interactive visualizations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

This project analyzes a telecom dataset of **7,000+ customer records** to:
- Identify key churn drivers (contract type, tenure, billing method)
- Build a **Logistic Regression model** achieving **82% accuracy**
- Deliver actionable business insights via dashboards

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── data/                        # Raw and processed datasets
│   ├── raw/                     # Original telecom dataset
│   └── processed/               # Cleaned & feature-engineered data
│
├── notebooks/                   # Jupyter Notebooks (EDA, modeling)
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Modeling.ipynb
│
├── src/                         # Python source modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluate.py
│
├── sql/                         # SQL queries for data analysis
│   └── churn_analysis.sql
│
├── reports/                     # Output charts and reports
│
├── tests/                       # Unit tests
│   └── test_preprocessing.py
│
├── main.py                      # Entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Aryansingh-B/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python main.py
```

---

## 📊 Key Results

| Metric        | Value  |
|---------------|--------|
| Accuracy      | 82%    |
| Precision     | 80%    |
| Recall        | 78%    |
| F1 Score      | 79%    |
| ROC-AUC       | 0.87   |

### Top Churn Drivers
1. **Month-to-month contracts** → 3x higher churn risk
2. **Fiber optic internet** → Higher churn than DSL
3. **Tenure < 12 months** → Most vulnerable segment
4. **Electronic check payment** → Correlated with churn

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data wrangling
- **Scikit-learn** — ML modeling
- **Matplotlib / Seaborn** — Visualizations
- **SQL (SQLite)** — Data querying
- **Tableau** — Business dashboard (see `/reports/`)

---

## 📈 Business Impact

- Identified **top churn segments** for targeted retention campaigns
- Enabled proactive **marketing interventions** for at-risk customers
- Simulated **15% churn reduction** via actionable recommendations

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
