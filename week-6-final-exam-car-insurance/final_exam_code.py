import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("car_insurance (1) (1).csv")

print("=== QUESTION 6: DATA CLEANING ===")
print(f"Missing values before imputation:")
print(f"  CREDIT_SCORE:   {df['CREDIT_SCORE'].isnull().sum()}")
print(f"  ANNUAL_MILEAGE: {df['ANNUAL_MILEAGE'].isnull().sum()}")

credit_median = df["CREDIT_SCORE"].median()
mileage_median = df["ANNUAL_MILEAGE"].median()
df["CREDIT_SCORE"] = df["CREDIT_SCORE"].fillna(credit_median)
df["ANNUAL_MILEAGE"] = df["ANNUAL_MILEAGE"].fillna(mileage_median)

print(f"\nImputation values used:")
print(f"  CREDIT_SCORE median:   {credit_median:.4f}")
print(f"  ANNUAL_MILEAGE median: {mileage_median:.2f}")
print(f"\nMissing values after imputation: {df.isnull().sum().sum()}")

le = LabelEncoder()
cat_cols = ["AGE", "GENDER", "DRIVING_EXPERIENCE", "EDUCATION", "INCOME", "VEHICLE_TYPE"]
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df = df.drop(columns=["ID", "POSTAL_CODE"])

X = df.drop(columns=["OUTCOME"]).astype(float)
y = df["OUTCOME"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== QUESTION 7: LOGISTIC REGRESSION SIGNIFICANT VARIABLES ===")
lr = LogisticRegression(max_iter=2000, random_state=42, C=1e10)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

X_arr = X_train.values
n = X_arr.shape[0]
X_b = np.column_stack([np.ones(n), X_arr])
coefs = np.concatenate([[lr.intercept_[0]], lr.coef_[0]])
proba = lr.predict_proba(X_arr)[:, 1]
W = proba * (1 - proba)
hessian = (X_b.T * W) @ X_b
cov = np.linalg.inv(hessian)
se = np.sqrt(np.diag(cov))
z = coefs / se
pvals = 2 * (1 - norm.cdf(np.abs(z)))

cols = ["const"] + list(X.columns)
res = pd.DataFrame({
    "Variable": cols,
    "Coefficient": np.round(coefs, 4),
    "P_Value": np.round(pvals, 4)
})
res = res[res["Variable"] != "const"]
sig = res[res["P_Value"] <= 0.05].sort_values("P_Value")
print("Significant variables at p <= 0.05:")
print(sig.to_string(index=False))

print("\n=== QUESTION 8: ENSEMBLE MODELS (1000 ITERATIONS) ===")
rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

gb = GradientBoostingClassifier(n_estimators=1000, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")
print(f"Random Forest Accuracy:       {rf_acc*100:.2f}%")
print(f"Gradient Boosting Accuracy:   {gb_acc*100:.2f}%")

print("\n=== QUESTION 9: CONFUSION MATRICES ===")
for name, preds in [("Logistic Regression", lr_pred), ("Random Forest", rf_pred), ("Gradient Boosting", gb_pred)]:
    cm = confusion_matrix(y_test, preds)
    print(f"\n{name}:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

print("\n=== QUESTION 10: FEATURE IMPORTANCE ===")
fi_rf = pd.DataFrame({
    "Feature": X.columns,
    "RF_Importance": np.round(rf.feature_importances_, 4)
}).sort_values("RF_Importance", ascending=False)
print("Random Forest Feature Importance:")
print(fi_rf.to_string(index=False))

fi_gb = pd.DataFrame({
    "Feature": X.columns,
    "GB_Importance": np.round(gb.feature_importances_, 4)
}).sort_values("GB_Importance", ascending=False)
print("\nGradient Boosting Feature Importance:")
print(fi_gb.to_string(index=False))
