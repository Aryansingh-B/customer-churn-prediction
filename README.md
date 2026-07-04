# Customer Churn Prediction & Intelligence Dashboard

An end-to-end machine learning project predicting telecom customer churn using XGBoost,
with a live business dashboard built in Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Live Demo
[View Dashboard](YOUR_STREAMLIT_URL_HERE)

---

## Business Problem

Telecom companies lose 15-25% of customers every year. Acquiring a new customer costs
5-7x more than retaining one. This project identifies at-risk customers before they churn
and quantifies the revenue impact — giving the business a clear, data-driven case for
targeted intervention.

---

## Key Features

- **Churn Prediction** — XGBoost classifier with ROC-AUC of ~0.84, trained on the IBM Telco dataset
- **Segment Analysis** — Churn breakdown by contract type, payment method, and subscribed services
- **Risk Scoring** — Low / Medium / High risk tiering assigned to every customer record
- **Live Predictor** — Enter any customer's details and get an instant churn probability score
- **Business KPIs** — Monthly revenue at risk, CLV impact estimates, and top retention levers

---

## Model Performance

| Metric    | Score |
|-----------|-------|
| ROC-AUC   | ~0.84 |
| F1 Score  | ~0.62 |
| Precision | ~0.67 |
| Recall    | ~0.57 |

---

## Key Business Insights

- **Month-to-month contracts** churn at 3x the rate of annual contracts — contract upgrade is the single biggest retention lever
- **Fiber optic customers** show the highest churn despite paying a premium, pointing to a service quality gap
- **Customers without online security** churn 35% more — cross-selling security services doubles as a retention tool
- **The first 12 months** is the critical retention window — churn risk drops sharply after year one

---

## Tech Stack

| Layer           | Tools                                        |
|-----------------|----------------------------------------------|
| Machine Learning | XGBoost, Scikit-learn, SMOTE               |
| Data Processing  | Pandas, NumPy                              |
| Visualization    | Plotly, Streamlit                          |
| Deployment       | Streamlit Cloud                            |

---

## Project Structure

````
customer-churn-prediction/
│
├── app.py
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── data/                  # NOT in repo
│   └── telco_churn.csv
├── models/                # NOT in repo (auto-generated)
│   ├── xgb_churn_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── test_predictions.csv
├── requirements.txt
├── setup.py
├── .env
├── .env.example
├── .gitignore
└── README.md
````

---

## Dataset Setup

The dataset is not included in this repo to keep it lightweight. Follow these steps:

- **Step 1** — Go to [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Step 2** — Download the ZIP (free Kaggle account required)
- **Step 3** — Unzip and place the CSV inside the `data/` folder
- **Step 4** — Rename it to `telco_churn.csv`

> The `data/` folder is gitignored and will never be accidentally pushed to GitHub.

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Aryansingh-B/customer-churn-prediction
cd customer-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env

# 5. Train the model
python -m src.model_training

# 6. Launch the dashboard
streamlit run app.py
```

---

## Deployment

This app auto-trains the model on first deploy via `setup.py` — no manual steps needed.
Just connect your GitHub repo to Streamlit Cloud and click Deploy.

---

## License

MIT