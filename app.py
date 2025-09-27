import streamlit as st
import pandas as pd
import joblib
import os

# === Load trained pipeline ===
model_path = os.path.join(os.path.dirname(__file__), "final_model.pkl")
model = joblib.load(model_path)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient health data below and get real-time prediction.")

# --- User Inputs ---
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST", [1, 2, 3])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [3, 6, 7])

# --- Prepare input DataFrame ---
input_data = pd.DataFrame([{
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 High Risk of Heart Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ No Heart Disease Detected (Probability: {prob:.2f})")
