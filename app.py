import streamlit as st
import pandas as pd
import joblib

# Load model + features
model = joblib.load("artifacts/model.pkl")
features = joblib.load("artifacts/feature_names.pkl")

st.title("Employee Attrition Risk - Responsible AI Demo")

# Sidebar input
st.sidebar.header("Employee Profile")
input_data = {}
for f in features:
    input_data[f] = st.sidebar.number_input(f, value=0.0)

# Prediction
if st.sidebar.button("Predict Risk"):
    X_input = pd.DataFrame([input_data])[features]
    prob = model.predict_proba(X_input)[0][1]
    risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    
    st.subheader("Prediction Results")
    st.write(f"Attrition Risk Score: **{prob:.2f}** → {risk}")

