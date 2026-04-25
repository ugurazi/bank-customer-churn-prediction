# Bank Customer Churn Prediction Dashboard

An end-to-end customer churn analysis and prediction project for the banking sector, combining machine learning modeling with an interactive Power BI dashboard.

## Project Overview

Customer churn is one of the most critical challenges in retail banking. This project aims to identify customers at risk of leaving, understand the key drivers behind churn, and provide actionable insights through a dynamic dashboard.

**Key Metrics at a Glance:**
- 10K customers analyzed
- 0.31 overall churn rate
- ~40% average risk score
- 3K high-risk customers identified

## What's Inside

- **Churn Prediction Model:** Binary classification model to predict whether a customer will churn, with risk scoring for each customer.
- **Customer Risk Analysis:** Segmentation of customers into Low / Medium / High risk categories based on model outputs.
- **Interactive Dashboard:** A multi-page Power BI dashboard featuring:
  - Executive Summary with KPIs
  - Customer Risk Analysis breakdown
  - Model Performance metrics
  - Dynamic filters by country, gender, credit card ownership, and activity status

## Key Insights

- Churn rates spike significantly for customers aged 40–60
- Germany has the highest churn rate (~33%) compared to Spain and France
- Nearly half of the customer base (48.68%) falls into the Low Risk category
- Inactive customers show considerably higher churn tendencies

## Tech Stack

- **Python** — Data preprocessing, feature engineering, model training
- **Scikit-learn / XGBoost** — Classification modeling
- **Power BI** — Interactive dashboard & data visualization
- **Pandas / NumPy** — Data manipulation

## Project Structure

```
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks (EDA, modeling)
├── models/                # Saved model artifacts
├── dashboard/             # Power BI (.pbix) file
└── README.md
```

## How to Use

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order for EDA and model training
4. Open the `.pbix` file in Power BI Desktop to explore the dashboard

## Contact

**Uğur Emir Azı**
- [LinkedIn](https://linkedin.com/in/uguremirazi)
- [GitHub](https://github.com/ugurazi)
