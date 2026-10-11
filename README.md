# Clinical Pathology & Machine Learning Analysis: Breast Cancer Diagnostic Project 🔬

## Overview
This project presents an exploratory data analysis (EDA) and predictive machine learning pipeline applied to the **Wisconsin Breast Cancer Diagnostic Dataset**. The primary goal is to use **Data Science** in **Medical Pathology** to extract useful insights from morphological and cytological cell features (extracted from Fine Needle Aspiration (FNA) biopsies) and accurately classify tumors as malignant or benign.

---

## Project Objectives
1. **Exploratory Data Analysis (EDA):** Investigate cytological differences (such as cell radius, perimeter, and concave points) between malignant and benign tissue samples.
2. **Statistical Correlation:** Identify the strongest histopathological predictors of tumor malignancy.
3. **Predictive Modeling:** Build a supervised machine learning classification model to predict tumor pathology with high clinical accuracy.

---

## Key Findings & Insights
1. **Morphological Variance:** The analysis revealed a statistically significant increase in cell size parameters (`mean radius`, `worst perimeter`) in malignant tumors compared to benign ones, which aligns perfectly with core pathology principles (nuclear pleomorphism and cellular enlargement).
2. **Strongest Predictors:** Correlation matrix analysis highlighted `worst concave points` (-0.79) and `mean concave points` (-0.78) as the most dominant numerical indicators associated with malignancy.
3. **Predictive Performance:** The Logistic Regression model achieved an overall **Accuracy of 95.61%**, demonstrating robust sensitivity and specificity in distinguishing tumor classes.

---

## Tech Stack & Libraries
- **Language:** Python 3.x
- **Data Manipulation & Analysis:** `Pandas`, `NumPy`
- **Data Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning & Metrics:** `Scikit-learn` (LogisticRegression, train_test_split, accuracy_score, classification_report)

---

## Visualizations

### 1. Boxplot Comparison
*Shows the distribution of cell mean radius across malignant vs. benign samples.*
<p align="center">
  <img width="686" height="547" alt="bp" src="https://github.com/user-attachments/assets/9e109281-baf9-4ffc-b884-1c7832f00955" />
</p>

### 2. Correlation Heatmap
*Visualizes feature-to-target correlations across all cytological parameters.*
<p align="center">
  <img width="1075" height="837" alt="hm" src="https://github.com/user-attachments/assets/3ce5e7b2-f26d-4500-8406-ef7dfcf54228" />
</p>

---

## Project Source Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

# Data exploration
df['target'] = cancer_data.target
df.head()
df.info()
df.shape
print(df[df.isnull().any(axis=1)])

print("\nTarget distribution (0: Malignant, 1: Benign):\n", df['target'].value_counts())

# EDA & Visualization 
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='mean radius', data=df, palette='Set2')
plt.title('Comparison of Cell Mean Radius by Tumor Pathology', fontsize=12, fontweight='bold')
plt.xlabel('Tumor Type (Malignant: 0, Benign: 1)', fontsize=10)
plt.ylabel('Mean Radius', fontsize=10)
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix[['target']], annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation of Features with Tumor Pathology Target', fontsize=14, fontweight='bold')
plt.show()

# Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
