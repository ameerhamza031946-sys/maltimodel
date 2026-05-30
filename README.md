# 📶 Multi-Model Machine Learning Comparison & Streamlit Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Data Science](https://img.shields.io/badge/Data%20Science-EDA%20%7C%20Modeling-success)

## 📌 Project Overview
This project is an industry-level Data Science application designed to predict customer churn in the telecommunications sector. It involves complete Exploratory Data Analysis (EDA), extensive feature engineering, and the training of 8 different machine learning algorithms. The final product includes an interactive Streamlit dashboard allowing users to explore data insights, compare models, and make real-time churn predictions.

## 🎯 Problem Statement
Customer churn is a critical issue for telecommunications companies. Losing customers results in lost revenue, and acquiring new customers is often more expensive than retaining existing ones. This project aims to build a reliable predictive model to identify high-risk customers so that targeted retention strategies can be applied.

## 📊 Dataset Information
* **Source:** Kaggle (Telco Customer Churn)
* **Description:** Contains information about a fictional telco company that provided home phone and internet services to 7043 customers in California in Q3.
* **Target Variable:** `Churn` (Whether the customer left within the last month or not).

## 📂 Project Structure
```
Telco-Customer-Churn-Project/
│
├── data/                       # Raw and processed datasets
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/                  # Jupyter notebooks for analysis
│   └── churn_analysis.ipynb
├── models/                     # Saved machine learning models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
├── screenshots/                # Dashboard screenshots
├── app.py                      # Streamlit application source code
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore
```

## ⚙️ Installation Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Telco-Customer-Churn-Project.git
   cd Telco-Customer-Churn-Project
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook (Optional to retrain models):**
   ```bash
   jupyter notebook notebooks/churn_analysis.ipynb
   ```

4. **Launch the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📈 Exploratory Data Analysis (EDA) Findings
* **Target Variable:** The dataset is imbalanced, with ~26.5% churn rate.
* **Contract Type:** Month-to-month contracts have a significantly higher churn rate compared to 1-year and 2-year contracts.
* **Payment Method:** Customers using electronic checks are much more likely to churn.
* **Tenure & Charges:** Customers with shorter tenure and higher monthly charges show higher churn propensity.

## 🧠 Machine Learning Models
We trained and evaluated the following models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K Nearest Neighbors
5. Support Vector Machine
6. Gradient Boosting
7. XGBoost
8. AdaBoost

## 🏆 Model Comparison Results
Models were compared based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC. 
* Ensemble methods (like Logistic Regression, Gradient Boosting and XGBoost) generally perform best, achieving high ROC AUC scores around ~0.84 to 0.85.
* The best model was automatically selected and saved as `best_model.pkl` for use in the prediction engine.

## 🚀 Deployment Instructions

### Option 1: Streamlit Community Cloud
1. Push this repository to your GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in.
3. Click **New app**, select your repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy**.

### Option 2: Render
1. Sign up on [Render](https://render.com/).
2. Create a **New Web Service** and connect your GitHub repository.
3. Use the following settings:
   * Build Command: `pip install -r requirements.txt`
   * Start Command: `streamlit run app.py --server.port $PORT`
4. Click **Create Web Service**.

## 🔮 Future Improvements
* Address class imbalance using SMOTE or ADASYN.
* Perform hyperparameter tuning using GridSearchCV or Optuna.
* Add explainable AI (XAI) features like SHAP values to the Streamlit app.

## 👨‍💻 Author Information
* **Name:** Multi-Model Data Scientist
* **Role:** Machine Learning Engineer / Data Analyst
