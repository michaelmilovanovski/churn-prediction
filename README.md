# Customer Churn Prediction

Predicting customer churn for a telecom company using machine learning.

## Problem Statement

Customer churn is costly—acquiring new customers is 5-25x more expensive than retaining existing ones. This project builds a predictive model to identify at-risk customers before they leave, enabling targeted retention efforts.

## Dataset

**Telco Customer Churn** (7,043 customers, 21 features)
- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Category | Features |
|----------|----------|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport, Streaming |
| Billing | MonthlyCharges, TotalCharges |
| Target | **Churn** (Yes/No) |

## Results

### Churn Distribution

![Churn Distribution](images/churn_distribution.png)

**26.5% of customers churned** — roughly 1 in 4 customers left the service.

### Model Performance

![Model Comparison](images/model_comparison.png)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.807** | **0.660** | **0.561** | **0.607** | **0.842** |
| Random Forest | 0.796 | 0.648 | 0.508 | 0.570 | 0.839 |
| XGBoost | 0.796 | 0.638 | 0.532 | 0.580 | 0.841 |

**Best Model:** Logistic Regression with ROC-AUC of 0.842

### ROC Curves

![ROC Curves](images/roc_curves.png)

All three models show strong discrimination ability with AUC scores above 0.83.

### Feature Importance

![Feature Importance](images/feature_importance.png)

### Churn by Contract Type

![Churn by Contract](images/churn_by_contract.png)

**Key Insight:** Month-to-month customers churn at **42.7%** compared to just **2.8%** for two-year contracts.

## Key Findings

1. **Churn Rate:** 26.5% of customers churned
2. **Contract Type:** Month-to-month customers have significantly higher churn rates
3. **Tenure Effect:** New customers (0-6 months) are most likely to leave
4. **Service Type:** Fiber optic internet customers churn more than DSL customers
5. **Payment Method:** Electronic check users have higher churn rates

## Business Recommendations

1. **Target new customers** with retention offers during the first 6 months
2. **Incentivize longer contracts** with discounts for 1-2 year commitments
3. **Investigate fiber optic service** — higher churn suggests possible quality issues
4. **Encourage automatic payments** — electronic check users churn more frequently
5. **Bundle security features** — customers with OnlineSecurity and TechSupport churn less

## Business Impact

Using this model on 1,409 test customers:
- Identified **210 of 374 churners** (56% recall)
- At $500 average customer value: **$52,500 potential savings** (assuming 50% retention rate)

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Train/test splits (cleaned)
├── notebooks/
│   ├── 01_exploration.ipynb     # EDA & visualizations
│   ├── 02_preprocessing.ipynb   # Data cleaning & feature engineering
│   └── 03_modeling.ipynb        # Model training & evaluation
├── images/                      # Visualization exports
├── README.md
└── requirements.txt
```

## Methodology

### 1. Exploratory Data Analysis
- Analyzed churn distribution (26.5% churn rate)
- Identified missing values (11 in TotalCharges)
- Visualized feature relationships with churn

### 2. Data Preprocessing
- Handled missing values (filled with 0 for new customers)
- Encoded categorical variables (one-hot encoding)
- Scaled numerical features (StandardScaler)
- Split data: 80% train / 20% test (stratified)

### 3. Model Training
- Logistic Regression (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting)

### 4. Evaluation
- Compared models using accuracy, precision, recall, F1-score, ROC-AUC
- Generated confusion matrices and ROC curves
- Analyzed feature importance

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook
```

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Visualization
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting

## Author

Michael Milovanovski

## License

This project is for educational purposes.
