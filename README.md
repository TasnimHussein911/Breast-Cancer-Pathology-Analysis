# Clinical Pathology & Machine Learning Analysis: Breast Cancer Diagnostic Project🔬

## Overview
This project presents an exploratory data analysis (EDA) and predictive machine learning pipeline applied to the **Wisconsin Breast Cancer Diagnostic Dataset**. The primary goal is to use **Data Science** in **Medical Pathology** to extract useful insights from morphological and cytological cell features (extracted from Fine Needle Aspiration (FNA) biopsies) and accurately classify tumors as malignant or benign.


## Project Objectives
1. **Exploratory Data Analysis (EDA):** Investigate cytological differences (such as cell radius, perimeter, and concave points) between malignant and benign tissue samples.
2. **Statistical Correlation:** Identify the strongest histopathological predictors of tumor malignancy.
3. **Predictive Modeling:** Build a supervised machine learning classification model to predict tumor pathology with high clinical accuracy.
 

## Key Findings & Insights
1. **Morphological Variance:** The analysis revealed a statistically significant increase in cell size parameters ('mean radius', 'worst perimeter') in malignant tumors compared to benign ones, which aligns perfectly with core pathology principles (nuclear pleomorphism and cellular enlargement).
2. **Strongest Predictors:** Correlation matrix analysis highlighted 'worst concave points' (-0.79) and 'mean concave points' (-0.78) as the most dominant numerical indicators associated with malignancy.
3. **Predictive Performance:** The Logistic Regression model achieved an overall **Accuracy of 95.61%**, demonstrating robust sensitivity and specificity in distinguishing tumor classes.
 

## Tech Stack & Libraries
- **Language:** Python 3.x
- **Data Manipulation & Analysis:** 'Pandas', 'NumPy'
- **Data Visualization:** 'Matplotlib', 'Seaborn'
- **Machine Learning & Metrics:** 'Scikit-learn' (LogisticRegression, train_test_split , accuracy_score , classification_report )


##  Visualizations

- **Boxplot Comparison:** Shows the distribution of cell mean radius across malignant vs. benign samples.
<img width="686" height="547" alt="bp" src="https://github.com/user-attachments/assets/9e109281-baf9-4ffc-b884-1c7832f00955" />

- **Correlation Heatmap:** Visualizes feature-to-target correlations across all cytological parameters.
<img width="1075" height="837" alt="hm" src="https://github.com/user-attachments/assets/3ce5e7b2-f26d-4500-8406-ef7dfcf54228" />

