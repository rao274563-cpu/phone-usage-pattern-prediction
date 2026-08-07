import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


#Loading on of the Best Model
with open("models/xgboost_model.pkl", "rb") as file:
    model = pkl.load(file)

# Label Encoder
with open("models/label_encoder.pkl", "rb") as file:
    label_encoder = pkl.load(file)    

# # Load Scaler
# with open("models/scaler.pkl", "rb") as file:
#     scaler = pkl.load(file)

# Load Feature Columns
with open("models/feature_columns.pkl", "rb") as file:
    feature_columns = pkl.load(file)



#Configuration
st.set_page_config(
    page_title="Phone Usage Pattern Prediction",
    page_icon="📱",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/phone_usage_india.csv")
    return df
df = load_data()

#st.title("📱 Phone Usage Pattern Prediction")
st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
Phone Usage Pattern Prediction
</h2>
""", unsafe_allow_html=True)

st.write(
    """
    Welcome to my Machine Learning Project.
    
    This application predicts a user's primary phone usage and also demonstrates clustering analysis performed on the dataset.
    """
)


#Slidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Explorer",
        "📈 EDA Dashboard",
        "🤖 Prediction",
        "🧩 Clustering",
        "📉 Model Comparison",
        "ℹ️ About"
    ]
)

# Home Page
if page == "🏠 Home":
    #st.title("📱 Phone Usage Pattern Prediction")
    st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
📱Phone Usage Pattern Prediction
</h2>
""", unsafe_allow_html=True)
    st.markdown("_ _ _")
    st.header("🎯 Project Objective")
    st.write(
        """
This project predicts the **Primary Phone Usage** of users using Machine Learning models.

The project also performs **Customer Segmentation (Clustering)** to identify different phone usage patterns."""
    )

    st.markdown("_ _ _")
    st.header("🚀 Technologies Used")
    st.write("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- Streamlit
- Matplotlib
- Seaborn
""")
    
    st.markdown("_ _ _")
    st.header("📚 Machine Learning Models")
    st.write("""
### Classification (Supervised Learning)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

### Clustering (Unsupervised Learning)
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model
""")
    
elif page == "📊 Dataset Explorer":
    #st.title("📊 Dataset Explorer")
    st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
📊 Dataset Explorer
</h2>
""", unsafe_allow_html=True)
    
    st.markdown("_ _ _")

    #Dataset shape
    st.subheader("📌Dataset Shape")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.markdown("_ _ _")

    #Dataset Preview
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head())
    
    st.markdown("_ _ _")

    # Datatype
    st.subheader("📋 Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.markdown("_ _ _")

    # Missing Value
    st.subheader("❓ Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing)

    st.markdown("_ _ _")

    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())

    st.markdown("_ _ _")

    # Download Dataset
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name="phone_usage_india.csv",
        mime="text/csv"
    )


elif page == "📈 EDA Dashboard"   :

    #st.title("📈 EDA Dashboard")
    st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
📈 Exploratory Data Analysis Dashboard
</h2>
""", unsafe_allow_html=True)
    
    st.markdown("_ _ _")

    chart = st.selectbox(
        "Selection Visualization",
        (
            "Age Distribution",
            "Gender Distribution",
            "Phone Brand Distribution",
            "Primary Use Distribution",
            "Screen Time Distribution",
            "Monthly Recharge Cost Distribution"
        )
    )

    st.markdown("_ _ _")

    if chart == "Age Distribution":
        fig, ax = plt.subplots(figsize=(7,4))

        ax.hist(df["Age"], bins=20)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    elif chart == "Gender Distribution":
        fig, ax = plt.subplots(figsize=(6,4))   

        df["Gender"].value_counts().plot(kind="bar", ax=ax)

        ax.set_title("Gender Distribution")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")

        st.pyplot(fig)

    elif chart == "Phone Brand Distribution":
        fig, ax = plt.subplots(figsize=(6,3))

        df["Phone Brand"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Phone Brand Distribution") 

        st.pyplot(fig)   

    elif chart == "Primary Use Distribution":
        fig, ax = plt.subplots(figsize=(6,3)) 

        df["Primary Use"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Primary Use Distribution")

        st.pyplot(fig)   

    elif chart == "Screen Time Distribution":
        fig, ax = plt.subplots(figsize=(6,3))

        ax.hist(df["Screen Time (hrs/day)"],bins=20)

        ax.set_title("Screen Time Distribution ")
        ax.set_xlabel("Hours per Day")

        st.pyplot(fig)   

    elif chart == "Monthly Recharge Cost Distribution":
        fig, ax = plt.subplots(figsize=(6,3))   

        ax.hist(df["Monthly Recharge Cost (INR)"], bins=20)

        ax.set_title("Monthly Recharge Cost Distribution")
        ax.set_xlabel("Monthly Recharge Cost")

        st.pyplot(fig)

elif page == "🤖 Prediction":
    #st.title("🤖 Prediction")
    st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
🤖 Prediction
</h2>
""", unsafe_allow_html=True)
    
    st.write("Enter the user's detail below.")

    age = st.number_input("Age", 10, 100, 25)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )
    
    location = st.selectbox(
        "Location",
        df["Location"].unique()
    )

    phone_brand = st.selectbox(
        "Phone Brand",
        df["Phone Brand"].unique()
    )

    operating_system = st.selectbox(
        "Operating System",
        df["OS"].unique()
    )

    screen_time = st.number_input(
        "Screen Time (hrs/day)",
        0.0,
        24.0,
        5.0
    )

    data_usage = st.number_input(
        "Data Usage (GB/month)",
        min_value = 1.0,
        max_value = 50.0,
        value = 5.0,
        step = 0.5
    )

    calls = st.number_input(
        "Calls Duration (mins/day)",
        min_value=5.0,
        max_value=300.0,
        value=150.0,
        step=5.0
    )

    apps = st.number_input(
        "Number of Apps installed",
        min_value=10,
        max_value=200,
        value = 100,
        step = 1
    )
    
    social = st.number_input(
        "Social Media Time (hrs/day)",
        0.0,
        24.0,
        2.0
    )

    ecommerce = st.number_input(
        "E-commerce Spend (INR/month)",
        0.0,
        100000.0,
        500.0
    )

    streaming = st.number_input(
        "Streaming Time (hrs/day)",
        0.0,
        24.0,
        1.0
    )

    gaming = st.number_input(
        "Gaming Time (hrs/day)",
        0.0,
        24.0,
        1.0
    )

    recharge = st.number_input(
        "Monthly Recharge Cost (INR)",
        min_value=100.0,
        max_value=2000.0,
        value=1000.0,
        step=50.0
    )

    predict = st.button("Predict")
    if predict:
        input_df = pd.DataFrame({
            "Age":[age],
            "Gender" : [gender],
            "Location" : [location],
            "Phone Brand" : [phone_brand],
            "OS" : [operating_system],
            "Screen Time (hrs/day)" : [screen_time],
            "Data Usage (GB/month)":[data_usage],
            "Calls Duration (mins/day)":[calls],
            "Number of Apps Installed":[apps],
            "Social Media Time (hrs/day)":[social],
            "E-commerce Spend (INR/month)":[ecommerce],
            "Streaming Time (hrs/day)":[streaming],
            "Gaming Time (hrs/day)":[gaming],
            "Monthly Recharge Cost (INR)":[recharge]

        })

        # One-Hot Encoding
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # Prediction
        prediction = model.predict(input_encoded)

        # Prediction Probability
        probabilities = model.predict_proba(input_encoded)

        # Highest confidence score
        confidence = np.max(probabilities) * 100

        # Convert numeric prediction back to label
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        st.success(f"🎯 Predicted Primary Use: **{predicted_label}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

elif page == "🧩 Clustering":
    #st.title("🧩 Clustering")
    st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
🧩 Clustering
</h2>
""", unsafe_allow_html=True)
    st.markdown("_ _ _")

    clustering_result = pd.DataFrame({
        "Algorithm":[
            "K-Means",
            "Hierarchical",
            "DBSCAN",
            "Gaussian Mixture"
        ],
        "Silhouette Score": [
            0.124708,
            0.060167,
            0.132411,
            0.059935

        ]
    })

    st.subheader("📊 Clustering Performance")
    st.dataframe(clustering_result, use_container_width=True)
    
    st.markdown("_ _ _")
    st.subheader("📈 Silhouette Score Comparison")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        clustering_result["Algorithm"],
        clustering_result["Silhouette Score"]
    )

    ax.set_xlabel("Clustering Algorithm")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Comparison of Clustering Algorithms")

    st.pyplot(fig)

    st.success("🏆 DBSCAN achieved the highest Silhouette Score and was selected as the best clustering model.")
    st.markdown("_ _ _")

    st.subheader("📝 Clustering Conclusion")
    st.write("""
    - Four Clustering Algorithms were evaluated.
    - DBSCAN achieved the highest Silhouette Score.
    - Therefore, DBSCAN selected as the final clustering algorithm.
    - The clustering analysis helps to identify users with similar smartphone usage pattern.
    """)
    

elif page == "📉 Model Comparison":
    #st.title("📉 Model Comparison")
    st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
📉 Model Comparison
</h2>
""", unsafe_allow_html=True)
    
    st.markdown("_ _ _")

    st.write("""
This section compares the performance of all classification models trained for predicting the user's Primary Phone Usage""")
    st.markdown("_ _ _")

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "LightGBM"
        ],
        "Accuracy": [
            0.1976,
            0.1979,
            0.1984,
            0.2027,
            0.1933
        ] 
    })

    st.subheader("📊 Model Performance")
    st.dataframe(model_results, use_container_width=True)

    st.markdown("_ _ _")
    st.subheader("📈 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        model_results["Model"],
        model_results["Accuracy"]
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Model Comparison")

    plt.xticks(rotation=20)

    st.pyplot(fig)
    

elif page == "ℹ️ About":
    #st.title("ℹ️ About")
    st.markdown("""
<h2 style='font-size:36px; font-weight:600; margin-bottom:10px;'>
ℹ️ About
</h2>
""", unsafe_allow_html=True)

    st.markdown("_ _ _")

    st.header("📱 Phone Usage Pattern Prediction")

    st.write("""
    This Machine Learning Project Predicts a User's **Primary Phone Usage**
    based on demographic and smartphone usage features.
    
    This project also performs **Customer Segmentation** using multiple clustering
    algorithms to identify different user behavior pattern.
    """)

    st.markdown("_ _ _")

    st.header("🎯 Project Features")

    st.write("""
    ✅ Data Exploration
    
    ✅ Exploratory Data Analysis
    
    ✅ Phone Usage Prediction

    ✅ Clustering
    
    ✅ Model Comparison
    
    ✅ Interactive Streamlit Dashboard
    """)

    st.markdown("_ _ _")

    st.header("🛠️ Technologies Used")

    st.write("""
    - Python
    - Pandas
    - Scikit-Learn
    - XGBoost
    - LightGBM
    - Streamlit
    - Matplotlib
    """)

    st.markdown("_ _ _")

    st.header("👨‍💻 Developed By")

    st.write("""
    **Sachin Kumar Rao**
    
    B.Tech - Computer Science Engineering
    
    Machine Learning | Data Science | Python
    """)