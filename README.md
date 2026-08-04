# 📱 Phone Usage Pattern Prediction using Machine Learning

## 📖 Project Overview

Phone Usage Pattern Prediction is an end-to-end Machine Learning project that predicts a user's **Primary Phone Usage** based on demographic information and smartphone usage behavior. The project also performs **customer segmentation using clustering algorithms** to discover groups of users with similar smartphone usage patterns.

An interactive **Streamlit web application** has been developed to allow users to explore the dataset, visualize insights, predict phone usage, compare machine learning models, and analyze clustering results.

---

## 🎯 Objectives

* Predict a user's primary phone usage category.
* Compare the performance of multiple Machine Learning algorithms.
* Perform customer segmentation using clustering techniques.
* Build an interactive Streamlit dashboard for prediction and data visualization.

---

## 📂 Dataset

The dataset contains smartphone usage information of users across India.

### Features

* Age
* Gender
* Location
* Phone Brand
* Operating System
* Screen Time (hrs/day)
* Data Usage (GB/month)
* Calls Duration (mins/day)
* Number of Apps Installed
* Social Media Time (hrs/day)
* E-commerce Spend (INR/month)
* Streaming Time (hrs/day)
* Gaming Time (hrs/day)
* Monthly Recharge Cost (INR)

### Target Variable

**Primary Use**

Classes:

* Education
* Entertainment
* Gaming
* Social Media
* Work

---

# 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib
* Seaborn
* Streamlit
* Pickle
* Git & GitHub

---

# 📊 Exploratory Data Analysis (EDA)

The project includes:

* Dataset overview
* Missing value analysis
* Duplicate value analysis
* Distribution analysis
* Histogram plots
* Box plots
* Correlation heatmap
* Feature-wise analysis

---

# ⚙️ Data Preprocessing

* Removed unnecessary columns
* One-Hot Encoding
* Label Encoding
* Train-Test Split
* Feature Alignment
* Model Serialization

---

# 🤖 Machine Learning Models

The following classification algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* LightGBM

---

# 📈 Model Performance

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression |     19.76% |
| Decision Tree       |     19.79% |
| Random Forest       |     19.84% |
| XGBoost             | **20.27%** |
| LightGBM            |     19.33% |

**Best Classification Model:** XGBoost

> **Note:** The comparatively low accuracy indicates limited predictive relationships within the available dataset rather than an implementation issue. Multiple algorithms were evaluated using the same preprocessing pipeline and produced similar performance.

---

# 🧩 Clustering Algorithms

The following clustering techniques were implemented:

* K-Means
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Model (GMM)

Evaluation was performed using the Silhouette Score.

---

# 🌐 Streamlit Application

The web application includes:

* 🏠 Home
* 📊 Dataset Explorer
* 📈 EDA Dashboard
* 🤖 Prediction
* 🧩 Clustering
* 📉 Model Comparison
* ℹ️ About

---

# 📁 Project Structure

```
PhoneUse_Pattern/
│
├── data/
│   └── phone_usage_india.csv
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── feature_columns.pkl
│   ├── label_encoder.pkl
│   ├── lighgb_model.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── xgboost_model.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_classification.ipynb
│   └── 05_clustering.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/rao274563-cpu/phone-usage-pattern-prediction.git
```

Move into the project directory:

```bash
cd phone-usage-pattern-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

* Improve prediction accuracy using a higher-quality dataset.
* Perform hyperparameter tuning.
* Add advanced feature engineering.
* Deploy the application to Streamlit Community Cloud.
* Integrate SHAP for model explainability.

---

# 👨‍💻 Author

**Sachin Kumar Rao**

B.Tech – Computer Science & Engineering

Machine Learning | Data Science | Python

LinkedIn: www.linkedin.com/in/sachin-rao-535b0b331
GitHub: https://github.com/rao274563-cpu

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
