import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# 1. Load data
df = pd.read_csv("loan_data.csv")

# 2. Select features & target
X = df[["person_age", "person_income", "loan_amnt"]]
y = df["loan_status"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train SVM model
model = SVC()
model.fit(X_train, y_train)

# 6. Save model
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Model trained and saved!")