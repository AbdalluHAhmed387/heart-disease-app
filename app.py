import streamlit as st
import pandas as pd
import joblib, os

# ========== Load Model ==========
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.pkl")
model = joblib.load(MODEL_PATH)

# ========== Page Config ==========
st.set_page_config(page_title="Heart Disease Predictor", page_icon="💓", layout="centered")

# ========== Custom CSS ==========
st.markdown("""
    <style>
    /* خلفية متدرجة */
    .stApp {
        background: linear-gradient(135deg, #1c1c1e, #2c2c2e, #3a3a3c);
        color: #f2f2f7;
    }

    /* كروت */
    .card {
        background-color: rgba(44, 44, 46, 0.9);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    }

    /* Result Box */
    .result-box {
        background-color: rgba(44, 44, 46, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin-top: 25px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    }

    /* زرار Predict */
    div.stButton > button:first-child {
        background-color: #ff4d6d;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #e04360;
        transform: scale(1.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9em;
        margin-top: 40px;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown("<h1 style='text-align:center;color:#ff4d6d'>💓 Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center'>Answer the following questions to estimate the risk of heart disease.</p>", unsafe_allow_html=True)

# ========== Inputs ==========
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    age = st.slider("👤 What is your age?", 18, 100, 40)
    sex = st.radio("⚧ Gender", ["Male", "Female"])
    cp = st.selectbox("❤️ Do you experience chest pain?", ["No pain", "Mild pain", "Moderate pain", "Severe pain"])
    trestbps = st.slider("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("🧪 Cholesterol Level (mg/dl)", 100, 400, 200)
    fbs = st.radio("🍬 Is your fasting blood sugar > 120 mg/dl?", ["No", "Yes"])
    restecg = st.selectbox("📉 Resting ECG Result", ["Normal", "Minor abnormality", "Heart muscle thickening"])
    thalach = st.slider("❤️ Max Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("🏃 Do you feel chest pain during exercise?", ["No", "Yes"])
    oldpeak = st.slider("📊 Exercise-related changes in ECG (ST Depression)", 0.0, 7.0, 1.0, 0.1)
    slope = st.selectbox("📈 Slope of ST Segment", ["Upsloping (normal)", "Flat", "Downsloping (concerning)"])
    ca = st.selectbox("🩻 Number of major blocked vessels", [0, 1, 2, 3])
    thal = st.selectbox("🧬 Thalassemia (blood disorder)", ["Normal", "Fixed Defect", "Reversible Defect"])
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Mapping ==========
sex = 1 if sex=="Male" else 0
cp = {"No pain":1, "Mild pain":2, "Moderate pain":3, "Severe pain":4}[cp]
fbs = 1 if fbs=="Yes" else 0
restecg = {"Normal":0, "Minor abnormality":1, "Heart muscle thickening":2}[restecg]
exang = 1 if exang=="Yes" else 0
slope = {"Upsloping (normal)":1, "Flat":2, "Downsloping (concerning)":3}[slope]
thal = {"Normal":3, "Fixed Defect":6, "Reversible Defect":7}[thal]

# ========== Create Input Data ==========
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]],
                          columns=["age","sex","cp","trestbps","chol","fbs","restecg",
                                   "thalach","exang","oldpeak","slope","ca","thal"])

# ========== Prediction ==========
if st.button("🔮 Predict Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#ff4d6d'>📊 Prediction Result</h2>", unsafe_allow_html=True)

    if prob is not None:
        st.write(f"Estimated risk of heart disease: **{prob*100:.1f}%**")

    if prediction == 1:
        st.error("⚠️ High Risk — Please consult a doctor as soon as possible.")
    else:
        st.success("✅ Low Risk — Keep maintaining a healthy lifestyle!")

    st.markdown('</div>', unsafe_allow_html=True)

# ================== Footer ==================
st.markdown("""
---
<div class="footer">
    <p>⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and should not replace professional medical advice.</p>
    <div style="margin-top:15px;">
        <a href="https://www.linkedin.com/in/abdalluhahmed387" target="_blank" style="margin-right:15px;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="35">
        </a>
        <a href="https://github.com/AbdalluHAhmed387" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="35">
        </a>
    </div>
    <p style="margin-top:10px;">🚀 Built with ❤ using <b>Python</b> & <b>Streamlit</b></p>
</div>
""", unsafe_allow_html=True)
