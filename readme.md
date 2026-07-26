# 🚀 Customer Churn Prediction & Business Intelligence System

A production-ready Machine Learning and Business Intelligence application that predicts customer churn, analyzes customer behavior, and provides interactive dashboards with actionable business insights.

The project demonstrates the complete Data Science lifecycle—from data preprocessing and exploratory data analysis to machine learning model development, explainability, business intelligence, reporting, and deployment.

---

# 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. Losing existing customers directly impacts revenue and business growth.

This project helps businesses identify customers who are likely to churn before they leave by using machine learning models and interactive business intelligence dashboards.

The application provides:

* Customer churn prediction
* Churn probability estimation
* Risk level classification
* Interactive business dashboards
* Business recommendations
* Revenue analysis
* Customer segmentation
* Downloadable reports

---

# 🎯 Objectives

* Predict customer churn accurately using Machine Learning.
* Analyze customer behavior using Business Intelligence dashboards.
* Identify high-risk customer segments.
* Estimate revenue loss caused by customer churn.
* Provide business recommendations for customer retention.
* Deploy a production-ready Streamlit application.

---

# 🛠 Technology Stack

| Category         | Technologies          |
| ---------------- | --------------------- |
| Programming      | Python                |
| Data Processing  | Pandas, NumPy         |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization    | Plotly                |
| Dashboard        | Streamlit             |
| Database         | SQLite                |
| Model Storage    | Joblib                |
| Version Control  | Git & GitHub          |

---

# 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── dashboard.py
├── predict.py
├── preprocessing.py
├── train_model.py
├── utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── prediction_history.db
│
├── models/
│   ├── preprocessing_pipeline.joblib
│   ├── feature_names.joblib
│   └── model.joblib
│
├── reports/
│   ├── model_evaluation_report.csv
│   ├── feature_importance.csv
│   ├── prediction_report.pdf
│   ├── analytics_report.pdf
│   └── preprocessing.log
│
├── assets/
├── notebooks/
├── requirements.txt
├── README.md
└── screenshots/
```

---

# 📊 Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer demographics, subscription details, billing information, contract types, internet services, payment methods, and customer churn status.

Target Variable:

```
Churn
```

Dataset Features include:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

---

# ⚙️ Project Workflow

```
Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning Models
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Prediction System
      │
      ▼
Business Intelligence Dashboard
      │
      ▼
Business Insights
      │
      ▼
Reports
      │
      ▼
Deployment
```

---

# 🧹 Phase 1 — Data Preprocessing

✔ Dataset validation

✔ Missing value handling

✔ Duplicate removal

✔ Outlier detection (IQR)

✔ Feature engineering

✔ Target encoding

✔ One-Hot Encoding

✔ Feature scaling

✔ Train/Test Split

✔ Preprocessing pipeline

✔ Processed dataset generation

---

# 📈 Phase 2 — Exploratory Data Analysis

Interactive visualizations include:

* Customer Distribution
* Churn Distribution
* Age Distribution
* Monthly Charges Analysis
* Contract Analysis
* Payment Method Analysis
* Correlation Heatmap
* Feature Relationships
* Business Insights

---

# 🤖 Phase 3 — Machine Learning

Implemented Models

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

The best-performing model is automatically selected based on ROC-AUC score.

---

# 🔍 Phase 4 — Model Explainability

Model explainability includes:

* Feature Importance
* Top Churn Drivers
* Business Interpretation
* Customer Risk Factors

Future Enhancement

* SHAP Explainability

---

# 🎯 Phase 5 — Prediction System

The prediction module provides:

* Customer Churn Prediction
* Churn Probability
* Risk Level
* Prediction Confidence
* Prediction History
* SQLite Storage

---

# 📊 Phase 6 — Business Intelligence Dashboard

Dashboard Features

### KPI Cards

* Total Customers
* Active Customers
* Churned Customers
* Churn Rate
* Average Monthly Revenue
* Estimated Revenue Loss

### Interactive Charts

* Customer Segmentation
* Churn by Contract
* Churn by Payment Method
* Monthly Charges Distribution
* Revenue Analysis
* Tenure Analysis
* Feature Importance
* Correlation Heatmap

### Interactive Filters

* Gender
* Contract Type
* Payment Method
* Internet Service
* Senior Citizen
* Tenure Range

---

# 💡 Phase 7 — Business Insights

Automatically generated insights include:

* High-risk customer segments
* Major churn drivers
* Revenue at risk
* Customer retention recommendations
* Executive summary

---

# 📄 Phase 8 — Reports

Users can download:

* Prediction Report (PDF)
* Analytics Report (PDF)
* Processed Dataset (CSV)

---

# ☁️ Phase 9 — Deployment

Deployment Platform

* Streamlit Cloud
* Render

---

# 📊 Machine Learning Pipeline

```
Raw Dataset

↓

Validation

↓

Cleaning

↓

Feature Engineering

↓

Encoding

↓

Scaling

↓

Train/Test Split

↓

Model Training

↓

Evaluation

↓

Model Selection

↓

Prediction

↓

Business Dashboard
```

---

# 📦 Installation

```bash
git clone https://github.com/your-username/customer-churn-prediction.git

cd customer-churn-prediction

pip install -r requirements.txt

python preprocessing.py

python train_model.py

streamlit run app.py
```

---

# 📈 Expected Outputs

* Trained Machine Learning Model
* Preprocessing Pipeline
* Feature Importance Report
* Model Evaluation Report
* Prediction Reports
* Interactive Dashboard
* Business Intelligence Insights

---

# 🌟 Key Features

* Production-ready architecture
* Modular Python code
* End-to-end Machine Learning pipeline
* Business Intelligence dashboard
* Interactive customer prediction system
* Automated reporting
* Explainable AI
* Professional project structure
* Deployment ready
* GitHub portfolio project

---

# 🚀 Future Improvements

* SHAP Explainability
* Automated Model Retraining
* Docker Support
* CI/CD Pipeline
* Cloud Database Integration
* REST API using FastAPI
* User Authentication
* Email Alerts for High-Risk Customers

---

# 👩‍💻 Author

**Kinza Arshad**

Data Science Undergraduate

Aspiring Data Engineer & AI Engineer

Passionate about Machine Learning, Business Intelligence, Data Analytics, and Production-Ready AI Solutions.

---

# ⭐ If you found this project useful, consider giving it a star on GitHub!
