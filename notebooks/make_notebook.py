import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """# Multi-Model Machine Learning Comparison
## Phase 1: Data Understanding"""
code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Display dataset shape
print(f"Dataset Shape: {df.shape}")

# 3. Show column names
print("\\nColumn Names:")
print(df.columns.tolist())

# 4. Display data types
print("\\nData Types:")
print(df.dtypes)

# 5. Show descriptive statistics
print("\\nDescriptive Statistics:")
display(df.describe())

# 6. Identify target variable
print("\\nTarget Variable is 'Churn'")

# 7. Check class distribution
print("\\nClass Distribution:")
print(df['Churn'].value_counts())

# 8. Generate data quality report
print("\\nMissing Values:")
print(df.isnull().sum())
"""

text_2 = """## Phase 2: Data Cleaning"""
code_2 = """# Replace spaces with NaN in TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Missing value handling
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Duplicate record handling
df.drop_duplicates(inplace=True)

# Drop CustomerID as it has no predictive power
df.drop('customerID', axis=1, inplace=True)

print("Data cleaning complete. Missing values remaining:")
print(df.isnull().sum().sum())
"""

text_3 = """## Phase 3: Exploratory Data Analysis"""
code_3 = """# 1. Target variable distribution
sns.countplot(x='Churn', data=df)
plt.title('Target Variable Distribution')
plt.show()

# 2. Gender analysis
sns.countplot(x='gender', hue='Churn', data=df)
plt.title('Gender vs Churn')
plt.show()

# 3. Senior citizen analysis
sns.countplot(x='SeniorCitizen', hue='Churn', data=df)
plt.title('Senior Citizen vs Churn')
plt.show()

# 4. Monthly charges analysis
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.show()

# 5. Tenure analysis
sns.violinplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs Churn')
plt.show()
"""

text_4 = """## Phase 4: Feature Engineering"""
code_4 = """from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encoding Target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn']) # Yes: 1, No: 0

# Encoding categorical features
cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Train-Test Split (80/20)
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
"""

text_5 = """## Phase 5 & 6: Machine Learning Models & Comparison"""
code_5 = """from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

results = []
best_model = None
best_auc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })
    
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model = model

results_df = pd.DataFrame(results).sort_values(by='ROC AUC', ascending=False)
display(results_df)

print(f"\\nBest Model selected based on ROC AUC: {results_df.iloc[0]['Model']}")

# Save the best model
joblib.dump(best_model, '../models/best_model.pkl')
print("Model saved to ../models/best_model.pkl")

# Save the scaler and feature columns for prediction
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(list(X.columns), '../models/columns.pkl')
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_code_cell(code_5)
]

with open('churn_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully.")
