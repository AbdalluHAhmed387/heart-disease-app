import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== Page Config ==================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="💓",
    layout="centered"
)

# ================== Custom CSS ==================
st.markdown("""
    <style>
    /* Dark background */
    .stApp {
        background-color: #1c1c1e;
        color: #f2f2f7;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 2.5em;
        color: #ff4d6d;
        margin-bottom: 5px;
    }
    .main-header p {
        font-size: 1.1em;
        color: #cccccc;
    }

    /* Cards for inputs */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #2c2c2e !important;
        border-radius: 10px;
        padding: 10px;
    }

    /* Prediction box */
    .result-box {
        background-color: #2c2c2e;
        padding: 20px;
        border-radius: 12px;
        margin-top: 25px;
        text-align: center;
    }
    .result-box h2 {
        color: #ff4d6d;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85em;
        color: #888;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown("""
<div class="main-header">
    <h1>💓 Heart Disease Risk Prediction</h1>
    <p>Enter patient details below to estimate the risk of heart disease</p>
</div>
""", unsafe_allow_html=True)

# ================== Input Form ==================
st.subheader("📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                      format_func=lambda x: {1:"Typical angina",2:"Atypical",3:"Non-anginal",4:"Asymptomatic"}[x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    restecg = st.selectbox("Resting ECG", [0, 1, 2],
                           format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"Hypertrophy"}[x])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)

slope = st.selectbox("Slope of ST segment", [1, 2, 3],
                     format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [3, 6, 7],
                    format_func=lambda x: {3:"Normal",6:"Fixed defect",7:"Reversible defect"}[x])

# ================== Collect Inputs ==================
input_data = pd.DataFrame({
    "age":[age], "sex":[sex], "cp":[cp], "trestbps":[trestbps], "chol":[chol],
    "fbs":[fbs], "restecg":[restecg], "thalach":[thalach], "exang":[exang],
    "oldpeak":[oldpeak], "slope":[slope], "ca":[ca], "thal":[thal]
})

# ================== Prediction ==================
if st.button("🔮 Predict Risk"):
    risk = np.random.rand()  # Replace with your model output

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("<h2>📊 Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(f"Estimated risk of heart disease: **{risk:.1%}**")

    if risk > 0.7:
        st.error("⚠️ High risk — Please consult a cardiologist immediately.")
    elif risk > 0.4:
        st.warning("⚠️ Moderate risk — Consider lifestyle improvements and checkups.")
    else:
        st.success("✅ Low risk — Keep maintaining a healthy lifestyle!")

    st.markdown("</div>", unsafe_allow_html=True)

    # Feature importance
    st.subheader("📈 Example Feature Importance")
    features = ["cp","thal","oldpeak","ca","age"]
    importance = [0.25,0.20,0.15,0.10,0.05]
    fig, ax = plt.subplots()
    ax.barh(features, importance, color="#ff4d6d")
    ax.set_xlabel("Importance")
    ax.set_facecolor("#1c1c1e")
    st.pyplot(fig)

# ================== Footer ==================
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
