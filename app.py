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
    background: linear-gradient(rgba(28,28,30,0.9), rgba(44,44,46,0.9)),
                url("https://img.freepik.com/free-vector/medical-healthcare-blue-background_1017-26807.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: #f2f2f7;
    font-family: "Segoe UI", sans-serif;
}



    /* العنوان */
    h1 {
        font-size: 2.2em;
        text-align: center;
        color: #ff4d6d;
    }

    /* الكروت */
    .card {
        background-color: rgba(44, 44, 46, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    }

    /* نتيجة */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin-top: 25px;
        text-align: center;
        font-size: 1.2em;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    }

    /* زرار */
    div.stButton > button:first-child {
        background-color: #ff4d6d;
        color: white;
        border: none;
        padding: 0.7em 1.5em;
        border-radius: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
        font-size: 1.1em;
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
st.markdown("<h1>💓 Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center'>Answer the following questions to estimate the risk of heart disease.</p>", unsafe_allow_html=True)


# ========== Guidelines ==========
with st.expander("📘 Guidelines"):
    st.markdown("""
    ### Feature Codes & Explanations

    | Column     | Description                                  | Values |
    |------------|----------------------------------------------|--------|
    | **age**    | Age of the person                           | 18–100 |
    | **sex**    | Gender                                      | 0 = Female, 1 = Male |
    | **cp**     | Chest Pain Type                             | 1 = No pain, 2 = Mild, 3 = Moderate, 4 = Severe |
    | **trestbps**| Resting Blood Pressure (mmHg)              | 80–200 |
    | **chol**   | Serum Cholesterol (mg/dl)                   | 100–400 |
    | **fbs**    | Fasting Blood Sugar > 120 mg/dl             | 0 = No, 1 = Yes |
    | **restecg**| Resting ECG Result                          | 0 = Normal, 1 = Minor abnormality, 2 = Heart muscle thickening |
    | **thalach**| Maximum Heart Rate Achieved                 | 60–220 |
    | **exang**  | Exercise-Induced Angina                     | 0 = No, 1 = Yes |
    | **oldpeak**| ST Depression (exercise-related ECG changes)| 0.0–7.0 |
    | **slope**  | Slope of ST Segment (ECG curve)             | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
    | **ca**     | Number of blocked major vessels             | 0–3 |
    | **thal**   | Thalassemia (blood disorder test result)    | 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect |
    """)

    st.info("""
    - **ECG (Electrocardiogram):** Test that measures the electrical activity of the heart.  
    - **ST Depression (oldpeak):** Drop in the ST segment of the ECG after exercise; higher values may indicate heart disease.  
    - **Slope of ST Segment:** Shape of the ST curve in ECG → Upsloping is usually normal, Flat/Downsloping may indicate issues.  
    - **Thalassemia (thal):** Blood disorder test result (Normal, Fixed Defect, or Reversible Defect).  
    - **Chest Pain (cp):** Type of chest pain experienced, which is an important indicator for heart problems.  
    """)

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

# ========== Batch Prediction from CSV ==========
st.markdown("### 📂 Upload CSV for Batch Prediction")

with st.expander("ℹ️ CSV Format Instructions"):
    st.markdown("""
    ✅ Your CSV file must include the following **columns**:

    - **age**: 18–100  
    - **sex**: 0 = Female, 1 = Male  
    - **cp**: 1 = No pain, 2 = Mild pain, 3 = Moderate pain, 4 = Severe pain  
    - **trestbps**: Resting BP (80–200 mmHg)  
    - **chol**: Cholesterol (100–400 mg/dl)  
    - **fbs**: 0 = No, 1 = Yes  
    - **restecg**: 0 = Normal, 1 = Minor abnormality, 2 = Heart muscle thickening  
    - **thalach**: Max Heart Rate (60–220)  
    - **exang**: 0 = No, 1 = Yes  
    - **oldpeak**: 0.0–7.0  
    - **slope**: 1 = Upsloping, 2 = Flat, 3 = Downsloping  
    - **ca**: 0–3  
    - **thal**: 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect  

    Example:

    ```
    age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
    52,1,3,130,180,0,0,170,0,0.0,1,0,3
    62,0,4,160,320,1,2,110,1,3.5,3,2,7
    45,1,2,120,200,0,1,150,0,1.2,2,0,6
    ```
    """)

uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    required_cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"]

    if not all(col in batch_data.columns for col in required_cols):
        st.error("❌ CSV file is missing required columns. Please follow the instructions above.")
    else:
        # Validation
        errors = []
        if batch_data["age"].min() < 18 or batch_data["age"].max() > 100:
            errors.append("❌ 'age' must be between 18 and 100.")
        if batch_data["trestbps"].min() < 80 or batch_data["trestbps"].max() > 200:
            errors.append("❌ 'trestbps' must be between 80 and 200.")
        if batch_data["chol"].min() < 100 or batch_data["chol"].max() > 400:
            errors.append("❌ 'chol' must be between 100 and 400.")
        if batch_data["thalach"].min() < 60 or batch_data["thalach"].max() > 220:
            errors.append("❌ 'thalach' must be between 60 and 220.")
        if batch_data["oldpeak"].min() < 0.0 or batch_data["oldpeak"].max() > 7.0:
            errors.append("❌ 'oldpeak' must be between 0.0 and 7.0.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            preds = model.predict(batch_data)
            probs = model.predict_proba(batch_data)[:,1] if hasattr(model, "predict_proba") else None

            results = batch_data.copy()
            results["Prediction"] = preds
            if probs is not None:
                results["Risk %"] = (probs * 100).round(1)

            st.dataframe(results)

            # Download
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", data=csv_out,
                               file_name="batch_predictions.csv", mime="text/csv")

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
