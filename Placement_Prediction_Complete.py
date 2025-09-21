# AIML Recruitment Task 2: Placement Prediction using Linear and Logistic Regression

## Student Information
- **Name**: [Your Name Here]
- **Student ID**: [Your Student ID]
- **Task**: Placement Prediction Models
- **Date**: September 2025

## Import Required Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


## Part A: Data Preprocessing

### Load and Explore Dataset


# Load the dataset
df = pd.read_csv('Placement_Prediction_dataset.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Description:")
print(df.describe())


### Data Quality Assessment


print("Missing Values:")
print(df.isnull().sum())
print("\nMissing value percentages:")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])

print("\nDuplicate Records:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

print("\nPlacement Statistics:")
print(f"Total Students: {len(df)}")
print(f"Students Placed: {df['Placed'].sum()}")
print(f"Placement Rate: {df['Placed'].mean():.2%}")


### Handle Missing Values


# Fill missing values with median for numerical columns
df['Sleep_Hours'].fillna(df['Sleep_Hours'].median(), inplace=True)
df['Internships'].fillna(df['Internships'].median(), inplace=True)
df['Projects'].fillna(df['Projects'].median(), inplace=True)

print("Missing values after treatment:")
print(df.isnull().sum())


### Handle Duplicates


# Remove duplicate rows
print(f"Before removing duplicates: {len(df)} rows")
df_clean = df.drop_duplicates()
print(f"After removing duplicates: {len(df_clean)} rows")
print(f"Removed {len(df) - len(df_clean)} duplicate rows")


### Handle Outliers


# Detect outliers using IQR method
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers

numerical_columns = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA', 'Placement_Score']

for col in numerical_columns:
    outliers = detect_outliers(df_clean[col])
    print(f"\n{col}: {len(outliers)} outliers detected")
    if len(outliers) > 0:
        print(f"Outlier range: {outliers.min():.2f} to {outliers.max():.2f}")


### Prepare Features for Modeling


# Prepare features and targets
feature_columns = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA']
X = df_clean[feature_columns]
y_regression = df_clean['Placement_Score']  # For Linear Regression
y_classification = df_clean['Placed']       # For Logistic Regression

print("Features prepared for modeling:")
print(f"Feature matrix shape: {X.shape}")
print(f"Regression target shape: {y_regression.shape}")
print(f"Classification target shape: {y_classification.shape}")


## Part B: Linear Regression - Predict Placement_Score

### Build Linear Regression Model


# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)

# Make predictions
y_pred_reg = lr_model.predict(X_test_reg_scaled)

print("Linear Regression Model Trained Successfully!")


### Evaluate Linear Regression Model


# Calculate performance metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Linear Regression Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (Linear Regression):")
print(feature_importance)


### Visualize Linear Regression Results


# Predicted vs Actual plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6, color='blue')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Placement Score')
plt.ylabel('Predicted Placement Score')
plt.title(f'Linear Regression: Predicted vs Actual\nR² = {r2:.3f}, RMSE = {rmse:.2f}')
plt.grid(True, alpha=0.3)
plt.show()

# Residual plot
residuals = y_test_reg - y_pred_reg
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_reg, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Placement Score')
plt.ylabel('Residuals')
plt.title('Residual Plot for Linear Regression')
plt.grid(True, alpha=0.3)
plt.show()

# Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Abs_Coefficient'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance - Linear Regression')
plt.grid(True, alpha=0.3)
plt.show()


## Part C: Logistic Regression - Predict Placement Status

### Build Logistic Regression Model


# Split data for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

# Scale features
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Train Logistic Regression model
log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
log_reg_model.fit(X_train_clf_scaled, y_train_clf)

# Make predictions
y_pred_clf = log_reg_model.predict(X_test_clf_scaled)
y_pred_proba = log_reg_model.predict_proba(X_test_clf_scaled)[:, 1]

print("Logistic Regression Model Trained Successfully!")


### Evaluate Logistic Regression Model


# Classification report
print("Logistic Regression Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))

# Confusion matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
print("\nConfusion Matrix:")
print(cm)

# Calculate additional metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print(f"\nAccuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")


### Visualize Logistic Regression Results


# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Placed', 'Placed'], 
            yticklabels=['Not Placed', 'Placed'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_clf, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Feature importance for logistic regression
log_feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': log_reg_model.coef_[0],
    'Abs_Coefficient': np.abs(log_reg_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(log_feature_importance['Feature'], log_feature_importance['Abs_Coefficient'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance - Logistic Regression')
plt.grid(True, alpha=0.3)
plt.show()


## Part D: Model Comparison and Insights

### Compare Model Performance


print("MODEL PERFORMANCE COMPARISON")
print("=" * 50)
print("\nLinear Regression (Placement Score Prediction):")
print(f"  R² Score: {r2:.3f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  Mean Absolute Error: {np.mean(np.abs(residuals)):.2f}")

print("\nLogistic Regression (Placement Status Prediction):")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1-Score: {f1:.3f}")
print(f"  AUC-ROC: {roc_auc:.3f}")


### Key Insights


print("\nKEY INSIGHTS FROM THE ANALYSIS")
print("=" * 50)

# Insight 1: Feature Importance Analysis
print("\nINSIGHT 1: MOST IMPORTANT FACTORS FOR PLACEMENT")
print("-" * 60)
print("Top factors for Placement Score (Linear Regression):")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {row['Feature']}: Coefficient = {row['Coefficient']:.3f}")

print("\nTop factors for Placement Status (Logistic Regression):")
for i, row in log_feature_importance.head(3).iterrows():
    print(f"  {row['Feature']}: Coefficient = {row['Coefficient']:.3f}")

# Insight 2: Model Effectiveness Comparison
print("\nINSIGHT 2: MODEL EFFECTIVENESS COMPARISON")
print("-" * 60)
if r2 > 0.7:
    print(f"Linear Regression performs WELL for predicting placement scores (R² = {r2:.3f})")
else:
    print(f"Linear Regression shows MODERATE performance for predicting placement scores (R² = {r2:.3f})")

if accuracy > 0.85:
    print(f"Logistic Regression performs EXCELLENTLY for predicting placement status (Accuracy = {accuracy:.3f})")
else:
    print(f"Logistic Regression shows GOOD performance for predicting placement status (Accuracy = {accuracy:.3f})")

# Insight 3: Student Performance Patterns
print("\nINSIGHT 3: STUDENT PERFORMANCE AND PLACEMENT READINESS")
print("-" * 60)

# Analyze high performers
high_cgpa = df_clean[df_clean['CGPA'] >= 8.5]
high_cgpa_placement_rate = high_cgpa['Placed'].mean()

# Analyze internship impact
with_internships = df_clean[df_clean['Internships'] >= 2]
internship_placement_rate = with_internships['Placed'].mean()

# Analyze project impact
high_projects = df_clean[df_clean['Projects'] >= 3]
project_placement_rate = high_projects['Placed'].mean()

print(f"Students with CGPA ≥ 8.5 have {high_cgpa_placement_rate:.1%} placement rate")
print(f"Students with ≥2 internships have {internship_placement_rate:.1%} placement rate")
print(f"Students with ≥3 projects have {project_placement_rate:.1%} placement rate")

overall_placement_rate = df_clean['Placed'].mean()
print(f"Overall placement rate: {overall_placement_rate:.1%}")


### Export Cleaned Dataset


# Export cleaned dataset
df_clean.to_csv('Placement_Prediction_dataset_cleaned.csv', index=False)
print("\nCleaned dataset exported as 'Placement_Prediction_dataset_cleaned.csv'")
print(f"Final dataset shape: {df_clean.shape}")


## Conclusion

This analysis demonstrates the application of both Linear and Logistic Regression for placement prediction:

1. **Linear Regression** effectively predicts placement scores with quantitative insights
2. **Logistic Regression** provides binary classification for placement status with high accuracy
3. **Key Success Factors**: CGPA, practical experience (internships), and project work are crucial
4. **Model Performance**: Both models show strong predictive capability for their respective targets
5. **Actionable Insights**: Students can focus on maintaining high CGPA, gaining internships, and completing projects

The analysis provides valuable insights for both students and institutions to improve placement outcomes.
