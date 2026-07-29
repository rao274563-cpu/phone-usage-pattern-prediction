# 📱 Phone Usage Pattern Prediction using Machine Learning

## 📌 Project Overview

This project predicts a user's **Primary Phone Usage** based on demographic information and smartphone usage behavior using Machine Learning. It also performs **Customer Segmentation** using multiple clustering algorithms to identify different smartphone usage patterns.

An interactive **Streamlit dashboard** allows users to explore the dataset, visualize data, make predictions, and compare machine learning models.

---

## 🎯 Project Objectives

* Predict the **Primary Phone Usage** of users.
* Compare multiple Machine Learning classification models.
* Perform customer segmentation using clustering algorithms.
* Build an interactive Streamlit web application for data exploration and prediction.

---

## 📂 Dataset

The dataset contains information about smartphone usage in India, including:

* Age
* Gender
* Location
* Phone Brand
* Operating System
* Screen Time
* Data Usage
* Call Duration
* Number of Apps Installed
* Social Media Time
* E-commerce Spend
* Streaming Time
* Gaming Time
* Monthly Recharge Cost
* Primary Phone Usage (Target)

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* LightGBM
* Streamlit
* Pickle

---

## 🧠 Machine Learning Models

### Classification Models

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost ✅ (Best Model)
* LightGBM

### Clustering Models

* K-Means ✅ (Best Clustering Model)
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Model (GMM)

---

## 📊 Exploratory Data Analysis

The project includes:

* Dataset Overview
* Missing Value Analysis
* Statistical Summary
* Age Distribution
* Gender Distribution
* Phone Brand Distribution
* Screen Time Distribution
* Monthly Recharge Cost Distribution
* Primary Usage Distribution

---

## 🤖 Prediction Pipeline

User Input

↓

One-Hot Encoding (`pd.get_dummies()`)

↓

Feature Alignment (`feature_columns.pkl`)

↓

StandardScaler

↓

XGBoost Model

↓

Predicted Primary Phone Usage

---

## 📈 Model Performance

| Model               | Accuracy        |
| ------------------- | --------------- |
| Logistic Regression | 0.1976          |
| Decision Tree       | 0.1979          |
| Random Forest       | 0.1984          |
| XGBoost             | **0.2027**      |
| LightGBM            | 0.1933          |

> **Best Classification Model:** XGBoost

---

## 🧩 Clustering Performance

| Algorithm        | Silhouette Score |
| ---------------- | ---------------- |
| K-Means          | 0.0728           |
| Hierarchical     | 0.0602           |
| DBSCAN           | 0.1324           |
| Gaussian Mixture | 0.0599           |

> **Best Clustering Model:** K-Means

---

## 📁 Project Structure

```text
Phone_Usage_Pattern_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── phone_usage_india.csv
│
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
│
└── notebooks/
```

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone <repository-link>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📌 Key Features

* Interactive Streamlit Dashboard
* Dataset Exploration
* Exploratory Data Analysis
* Machine Learning Prediction
* Clustering Analysis
* Model Comparison
* Professional Dashboard Interface

---

## 🔮 Future Improvements

* Improve model performance through feature engineering and hyperparameter tuning.
* Add additional visualizations and interactive charts.
* Integrate real-time data collection.
* Deploy using Docker and cloud platforms.

---

## 👨‍💻 Author

**Sachin Kumar Rao**

B.Tech – Computer Science & Engineering

Machine Learning | Data Science | Python

LinkedIn: www.linkedin.com/in/sachin-rao-535b0b331
GitHub: https://github.com/rao274563-cpu

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
