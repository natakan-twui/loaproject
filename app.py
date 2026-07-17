import streamlit as st
import pandas as pd
import numpy as np
import joblib

# โหลดโมเดล
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🏦 Loan Prediction")

age = st.number_input("Age", 18, 100, 25)
income = st.number_input("Income", 1000, 1000000, 30000)
loan = st.number_input("Loan Amount", 500, 100000, 10000)

if st.button("Predict"):
    data = pd.DataFrame(
        [[age, income, loan]],
        columns=["person_age", "person_income", "loan_amnt"]
    )

    data = pd.get_dummies(data)
    data = data.reindex(columns=columns, fill_value=0)

    data_scaled = scaler.transform(data)

    result = model.predict(data_scaled)

    if result[0] == 1:
        st.success("Approved")
    else:
        st.error("Not Approved")