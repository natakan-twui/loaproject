import streamlit as st
import pandas as pd
import joblib


# =========================
# โหลดโมเดล
# =========================

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# =========================
# ตั้งค่าหน้าเว็บ
# =========================

st.set_page_config(
    page_title="ระบบทำนายอนุมัติสินเชื่อ",
    page_icon="🏦",
    layout="centered"
)


# =========================
# CSS ตกแต่งเว็บ
# =========================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .title {
        text-align:center;
        color:#1f4e79;
        font-size:35px;
        font-weight:bold;
    }

    .box {
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 0px 10px #cccccc;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# =========================
# หัวเว็บ
# =========================

st.markdown(
    """
    <div class="title">
    🏦 ระบบทำนายการอนุมัติสินเชื่อด้วย AI
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    "กรอกข้อมูลด้านล่างเพื่อให้โมเดล SVM วิเคราะห์โอกาสอนุมัติสินเชื่อ"
)


st.divider()



# =========================
# รับข้อมูล
# =========================

age = st.slider(
    "👤 อายุ",
    min_value=18,
    max_value=100,
    value=25
)


income = st.number_input(
    "💰 รายได้ต่อปี",
    min_value=1000,
    max_value=1000000,
    value=30000
)


loan = st.number_input(
    "💳 จำนวนเงินกู้",
    min_value=500,
    max_value=1000000,
    value=10000
)



st.divider()



# =========================
# ทำนายผล
# =========================

if st.button("🔍 ทำนายผล"):


    # สร้างข้อมูลให้ตรงกับตอน Train

    data = pd.DataFrame(
        {
            "person_age": [age],
            "person_income": [income],
            "loan_amnt": [loan]
        }
    )


    # Standardization

    data_scaled = scaler.transform(data)


    # ทำนาย

    result = model.predict(data_scaled)


    st.subheader("📌 ผลการวิเคราะห์")


    if result[0] == 1:

        st.success(
            "✅ มีโอกาสได้รับการอนุมัติสินเชื่อ"
        )

    else:

        st.error(
            "❌ มีโอกาสไม่ได้รับการอนุมัติสินเชื่อ"
        )