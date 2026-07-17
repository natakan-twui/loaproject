import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดล
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ตั้งค่าเว็บ
st.set_page_config(page_title="Loan Prediction", layout="centered")

# HEADER
st.markdown("<h1 style='text-align: center;'>🏦 Loan Approval Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# INPUT SECTION
st.subheader("📋 Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", 18, 100, 25)
    income = st.number_input("💰 Income", 1000, 1000000, 30000)

with col2:
    loan = st.number_input("🏦 Loan Amount", 500, 100000, 10000)

st.markdown("---")

# BUTTON
if st.button("🔍 Predict", use_container_width=True):

    data = pd.DataFrame(
        [[age, income, loan]],
        columns=["person_age", "person_income", "loan_amnt"]
    )

    data = pd.get_dummies(data)
    data = data.reindex(columns=columns, fill_value=0)
    data_scaled = scaler.transform(data)

    result = model.predict(data_scaled)

    st.markdown("### 📊 Result")

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

# FOOTER
st.markdown("---")
st.caption("Developed using SVM & Streamlit")