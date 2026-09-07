# BH-AI-ML-Assignment-17.1
This project compares K-Nearest Neighbors, Logistic Regression, Decision Tree, and Support Vector Machine classifiers to predict whether a customer will subscribe to a term deposit, using data from a Portuguese banking institution's telemarketing campaigns.

Notebook: github.com/RaniDeepLearning/BH-AI-ML-Assignment-17.1/blob/main/term_deposit_pred.ipynb

Data source: UCI Machine Learning Repository – Bank Marketing Dataset

# Business Objective
Aims at Predicting which customers are more likely to subscribe to a term deposit so the bank can prioritise outreach, improve telephone marketing efficiency, reduce unnecessary calls, and use time and resources more effectively.

# Data
  - 41,188 records, 21 columns, spanning 17 marketing campaigns (May 2008 – November 2010).
  - No missing values, but several categorical columns contain an explicit "unknown" category.
  - Target is highly imbalanced: only 11.3% of customers subscribed.
  - duration (call length) was excluded from modelling since it's only known after a call takes place and would cause data
    leakage.

# Methodology
1. EDA — distribution checks, correlation heatmap, violin/bar plots comparing subscribers vs. non-subscribers.
2. Feature engineering — modeled using only bank client features (age, job, marital, education, default, housing, loan), encoded  via a ColumnTransformer (median imputation + scaling for numeric, most-frequent imputation + one-hot encoding for categorical).
3. Train/test split — 80/20, stratified to preserve class balance.
4. Baseline — majority-class DummyClassifier, 88.7% accuracy but 0% recall on subscribers.
5. Model comparison — Logistic Regression, KNN, Decision Tree, and Linear SVM evaluated with stratified k-fold cross-validation     across multiple metrics (ROC-AUC, average precision, precision, recall, F1), not just accuracy.
6. Hyperparameter tuning — GridSearchCV optimizing average precision, including class_weight options to address imbalance.
7. Threshold tuning — optimal decision thresholds selected per model to balance precision/recall, since imbalance renders the       default 0.5 threshold ineffective.
8. Interpretation — logistic regression coefficients and odds ratios examined to identify which customer segments are more/less     likely to subscribe.

# Key Findings
  - The baseline's 88.7% accuracy is misleading — it comes from predicting every customer as a non-subscriber and catches zero       actual subscribers.
  - The tuned Decision Tree had the highest average precision (0.211) and ROC-AUC (0.663), meaning it ranks likely subscribers       best overall, but its default-threshold recall was only 3.9%.
  - Logistic Regression was selected as the practical choice: easier to interpret, and after threshold tuning (threshold = 0.12) 
    it achieved:<br>
      Precision: 17.0%<br>
      Recall: 52.3%<br>
      F1-score: 25.6%<br>
  - The model correctly identified 485 subscribers but also produced 2,373 false positives.
  - Students, retired, and single customers were more likely to subscribe; blue-collar, service, and entrepreneurial workers,and     customers with unknown default status, were less likely to subscribe.
  - These are associations, not causal relationships.

# Findings:
- Only 11.3% of customers subscribed, so the target was highly imbalanced.
- The baseline accuracy was 88.7%, but the baseline model did not identify any subscribers.
- The Decision Tree had the highest cross-validation average precision of 0.215 and a test average precision of 0.210.
- The Decision Tree and Linear SVM both had a test ROC-AUC of about 0.652.
- After threshold tuning, Linear SVM had the highest recall of 50.0% and the highest F1-score of 25.9%.
- I used Logistic Regression for detailed interpretation because it was easier to explain.
- At a threshold of 0.14, Logistic Regression had 18.8% precision, 34.7% recall, and an F1-score of 24.4%.
- Logistic Regression correctly identified 322 subscribers but incorrectly identified 1,388 non-subscribers as subscribers.
- Students, retired customers, and single customers showed a higher likelihood of subscribing.
- These results show patterns in the data but do not prove that these features caused customers to subscribe.

# Recommendations:
- The bank should contact customers with the highest predicted subscription probabilities first.<br>
- The decision threshold should be adjusted based on how many customers the bank can contact.<br>
- A higher threshold can reduce unnecessary calls, but it may also miss some subscribers.<br>
- More customer and previous campaign information may help improve the model.<br>
- Call duration should not be used for advance predictions because it is only known after the call.<br>
- The bank should first test the model on a small campaign and check the results before using it on a larger scale.<br>

# Next Steps
- Test the Logistic Regression model on a small marketing campaign.<br>
- Select a final threshold based on the bank’s budget and calling capacity.<br>
- Collect more customer information that is available before the call.<br>
- Compare the results with other models, such as Random Forest or Gradient Boosting.<br>
- Check whether the model performs fairly across different customer groups.<br>
- Monitor precision and recall over time and retrain the model when performance decreases.<br>

# Conclusion
In this project, I compared four models to predict whether a customer would subscribe to a term deposit. Since the data was imbalanced, I focused on precision, recall, F1-score, and average precision instead of accuracy alone. Linear SVM had the highest recall and F1-score after threshold tuning, while the Decision Tree had the highest average precision. I used Logistic Regression for detailed interpretation because it was easier to explain. At the selected threshold, it identified about 35% of subscribers with a precision of about 19%. The model can help the bank rank customers for future calls, but it should be tested before being used in a real campaign.

# Author
Rani Kumari

PCAIML Berkley Haas
