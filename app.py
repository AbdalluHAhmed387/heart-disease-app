import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== Page Config ==================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Custom CSS ==================
st.markdown("""
    <style>
    /* App background */
    .stApp {
        background-color: #f9f9f9;
    }

    /* Header style */
    .main-header {
        background: linear-gradient(90deg, #b30000, #e63946);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
    }
    .main-header h1 {
        font-size: 2.2em;
        margin: 0;
    }
    .main-header p {
        margin: 0;
        font-size: 1.1em;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #b30000;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #e60000;
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f2f2f2;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85em;
        color: #555;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown("""
<div class="main-header">
    <h1>💓 Heart Disease Risk Prediction</h1>
    <p>Predict your risk of heart disease using health parameters</p>
</div>
""", unsafe_allow_html=True)

# ================== Sidebar Inputs ==================
st.sidebar.header("📝 Input Patient Data")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", options=[0,1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[1,2,3,4],
                          format_func=lambda x: {1:"Typical angina",2:"Atypical",3:"Non-anginal",4:"Asymptomatic"}[x])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
restecg = st.sidebar.selectbox("Resting ECG", options=[0,1,2],
                               format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"Hypertrophy"}[x])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
slope = st.sidebar.selectbox("Slope of ST segment", options=[1,2,3],
                             format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", options=[0,1,2,3])
thal = st.sidebar.selectbox("Thalassemia (thal)", options=[3,6,7],
                            format_func=lambda x: {3:"Normal",6:"Fixed defect",7:"Reversible defect"}[x])

# Collect inputs
input_data = pd.DataFrame({
    "age":[age], "sex":[sex], "cp":[cp], "trestbps":[trestbps], "chol":[chol],
    "fbs":[fbs], "restecg":[restecg], "thalach":[thalach], "exang":[exang],
    "oldpeak":[oldpeak], "slope":[slope], "ca":[ca], "thal":[thal]
})

# ================== Prediction ==================
if st.sidebar.button("🔮 Predict"):
    # placeholder model
    risk = np.random.rand()  # replace with: model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    st.markdown(f"Estimated risk of heart disease: **{risk:.1%}**")

    if risk > 0.7:
        st.error("⚠️ High risk — Please consult a cardiologist immediately.")
    elif risk > 0.4:
        st.warning("⚠️ Moderate risk — Consider lifestyle improvements and medical checkups.")
    else:
        st.success("✅ Low risk — Keep maintaining a healthy lifestyle!")

    # Feature importance example
    st.subheader("📈 Example Feature Importance")
    features = ["cp","thal","oldpeak","ca","age"]
    importance = [0.25,0.20,0.15,0.10,0.05]
    fig, ax = plt.subplots()
    ax.barh(features, importance, color="#b30000")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ================== Footer ==================
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
