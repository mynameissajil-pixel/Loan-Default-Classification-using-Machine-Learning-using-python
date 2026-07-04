# Loan Default Classification using Machine Learning | Python

A machine learning project that predicts whether a borrower will default on a loan, using classification models trained on borrower financial and demographic data.

## 📌 Introduction

This project builds and compares multiple classification models to predict loan default risk. The goal is to identify borrowers likely to default and uncover the key financial and demographic factors driving that risk, supporting better lending decisions.

## 🎯 Objective

- Build and compare classification models (Logistic Regression, Random Forest, Decision Tree) to predict loan default.
- Handle class imbalance in the target variable using SMOTE.
- Identify key risk-driving factors through EDA and correlation analysis.
- Tune models using GridSearchCV for improved performance.

## 🗂️ Dataset

- **Source:** `Loan_default.csv`
- **Records:** 43,121 loan records, 18 columns
- **Features:** Age, Income, Loan Amount, Credit Score, Months Employed, Number of Credit Lines, Interest Rate, Loan Term, DTI Ratio, Education, Employment Type, Marital Status, Has Mortgage, Has Dependents, Loan Purpose, Has Co-Signer
- **Target variable:** `Default` (binary: 0 = No Default, 1 = Default)

## 🛠️ Tools & Libraries

- **Python**
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — data visualization
- **Scikit-learn** — model building, tuning, and evaluation
- **imbalanced-learn (SMOTE)** — class imbalance handling

## 🔍 Workflow / Methodology

1. **Data Cleaning** — removed missing values and duplicate rows
2. **EDA** — visualized income distribution, age distribution, marital status vs. default, and feature correlation heatmap
3. **Feature Engineering** — label encoding (binary features), one-hot encoding (categorical features), ordinal mapping (Education)
4. **Handled Class Imbalance** — applied SMOTE, balancing the training set to 30,433 records per class
5. **Feature Scaling** — StandardScaler applied to training and test sets
6. **Model Training** — Logistic Regression, Random Forest, Decision Tree
7. **Hyperparameter Tuning** — GridSearchCV applied to Logistic Regression and Decision Tree
8. **Evaluation** — accuracy score, confusion matrix, classification report

## 📊 Exploratory Data Analysis

### Correlation Heatmap
![Correlation Heatmap](screenshots/correlation-heatmap.png)

### Loan Default Distribution
![Default Distribution](screenshots/default-distribution.png)

### Marital Status vs Default
![Marital Status vs Default](screenshots/marital-status-vs-default.png)

## 🤖 Model Training & Results

### Logistic Regression
```python
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
```
**Accuracy:** 80.91%

**Confusion Matrix:**
```
[[6635 1024]
 [ 622  343]]
```

**Classification Report:**
```
              precision    recall  f1-score   support

         0.0       0.91      0.87      0.89      7659
         1.0       0.25      0.36      0.29       965

    accuracy                           0.81      8624
   macro avg       0.58      0.61      0.59      8624
weighted avg       0.84      0.81      0.82      8624
```

After GridSearchCV tuning:
```
Best parameters: {'C': 0.01, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}
Best cross-validation accuracy: 84.35%
Test set accuracy: 80.95%
```

### Random Forest
```python
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(x_train, y_train)
pred = RF.predict(x_test)
```
**Accuracy:** 85.61%

### Decision Tree
```python
treee = DecisionTreeClassifier(criterion="gini", max_depth=5)
treee.fit(x_train, y_train)
```
**Accuracy:** 68.55%

## 📈 Results Summary

| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | 80.91% |
| Logistic Regression (GridSearchCV tuned) | 80.95% |
| Random Forest | 85.61% |
| Decision Tree | 68.55% |

## 💡 Key Insights

- Random Forest achieved the highest test accuracy (85.61%), outperforming both Logistic Regression and Decision Tree.
- GridSearchCV tuning on Logistic Regression improved cross-validation accuracy to 84.35%, but test accuracy only moved from 80.91% to 80.95% — a marginal real-world gain.
- Decision Tree (max depth = 5) underperformed significantly (68.55%), suggesting it was too shallow to capture the patterns in this dataset.
- The original dataset was imbalanced toward non-default cases; SMOTE was used to balance the training data (30,433 records per class) before model training.

## 🎯 Conclusion

Random Forest was the best-performing model for predicting loan default in this dataset, offering the highest accuracy among the three approaches tested. While Logistic Regression is simpler and faster to train, Random Forest's ability to capture non-linear relationships between borrower features made it more effective at identifying default risk.

## 📓 Full Notebook

The complete code, outputs, and visualizations are available in the notebook:
[loan_default.ipynb](https://github.com/mynameissajil-pixel/Loan-Default-Classification-using-Machine-Learning-using-python/blob/043f8d4822648d3ddbccadd7e8e31e4fafc071da/loan_default.ipynb)

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python loan_default.py
```

## 📄 License

*[Optional — e.g., "For portfolio/educational purposes only"]*

## 📬 Contact

*[Your name / LinkedIn / email]*
