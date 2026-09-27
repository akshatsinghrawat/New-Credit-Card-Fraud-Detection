# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using imbalanced classification techniques and multiple machine learning models.

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small proportion of all transactions.

The objective of this project is to build machine learning models that can identify fraudulent transactions while minimizing false alerts on legitimate transactions.

The project focuses on proper handling of class imbalance, leakage-free preprocessing, and evaluation using metrics that are more meaningful than accuracy.

## Dataset

The project uses the Credit Card Fraud Detection dataset containing transactions made by European cardholders.

The original dataset contains:

- 284,807 transactions
- 31 features
- 492 fraudulent transactions
- `Time` and `Amount` transaction-related features
- `V1`–`V28`: anonymized PCA-transformed features
- `Class`: target variable
  - `0` = Legitimate
  - `1` = Fraud

After removing duplicate transactions, the dataset used for modeling contained 283,726 observations and 473 fraudulent transactions.

### Dataset Source

Kaggle:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset is not included in this repository because of its large size.

## Methodology

The project follows the following workflow:

1. Data loading and initial inspection
2. Duplicate detection and removal
3. Exploratory Data Analysis
4. Transaction amount and time analysis
5. Feature correlation analysis
6. Stratified train-test split
7. Feature scaling using StandardScaler
8. Class imbalance handling using SMOTE
9. Model training
10. Model evaluation
11. Confusion matrix analysis
12. ROC curve analysis
13. Precision-Recall curve analysis
14. Random Forest feature importance analysis

### Handling Class Imbalance

The dataset contains significantly fewer fraudulent transactions than legitimate transactions.

SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training data to increase the representation of the minority class.

The test set was kept untouched to provide an unbiased evaluation of model performance.

## Machine Learning Models

Three classification models were evaluated:

- Logistic Regression
- Random Forest
- HistGradientBoosting

## Model Performance

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1077 | 0.8737 | 0.1917 | 0.9646 | 0.7098 |
| Random Forest | 0.8111 | 0.7684 | 0.7892 | 0.9769 | 0.7998 |
| HistGradientBoosting | 0.6311 | 0.8105 | 0.7097 | 0.9739 | 0.7983 |

### Key Observations

- Logistic Regression achieved the highest recall at **87.37%**, but generated substantially more false positives.
- Random Forest achieved **81.11% precision**, **76.84% recall**, and **78.92% F1-score**.
- Random Forest achieved the highest ROC-AUC of **97.69%** and PR-AUC of **79.98%** among the evaluated models.
- HistGradientBoosting achieved **81.05% recall** and **79.83% PR-AUC**, providing competitive performance.
- The results demonstrate the importance of balancing fraud detection with false-positive control rather than relying on accuracy alone.

## Feature Importance

Random Forest feature importance was used to identify the most influential features in the model.

The top features included:

- V14
- V10
- V12
- V17
- V4

Since the `V1`–`V28` variables are anonymized PCA-transformed features, their importance cannot be directly interpreted as specific real-world transaction characteristics.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── CreditCardFraudDetection_v2.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore