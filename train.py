import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


# โหลดข้อมูล
df = pd.read_csv("loan_data.csv")


# ใช้แค่ 3 feature
X = df[
    [
        "person_age",
        "person_income",
        "loan_amnt"
    ]
]

y = df["loan_status"]


# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)


# Train SVM
model = SVC()

model.fit(
    X_train,
    y_train
)


# save model

joblib.dump(
    model,
    "svm_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)


# save columns
joblib.dump(
    [
        "person_age",
        "person_income",
        "loan_amnt"
    ],
    "columns.pkl"
)


print("สร้างโมเดลสำเร็จ")