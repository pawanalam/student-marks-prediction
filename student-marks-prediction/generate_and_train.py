"""
Generate synthetic student dataset and train a Linear Regression model.
Run this once before starting the Flask app:  python generate_and_train.py
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
N = 1000

study_hours      = np.random.uniform(0, 12, N)          # hours/day
attendance       = np.random.uniform(40, 100, N)         # percentage
prev_score       = np.random.uniform(30, 100, N)         # previous exam score
sleep_hours      = np.random.uniform(3, 10, N)           # hours/night
assignments_done = np.random.uniform(0, 100, N)          # % assignments completed

# Target: weighted combination + small noise
marks = (
    study_hours      * 3.5  +
    attendance       * 0.25 +
    prev_score       * 0.40 +
    sleep_hours      * 1.20 +
    assignments_done * 0.15 +
    np.random.normal(0, 4, N)
)
marks = np.clip(marks, 0, 100)

df = pd.DataFrame({
    "study_hours":      study_hours,
    "attendance":       attendance,
    "prev_score":       prev_score,
    "sleep_hours":      sleep_hours,
    "assignments_done": assignments_done,
    "marks":            marks,
})

os.makedirs("model", exist_ok=True)
df.to_csv("model/student_data.csv", index=False)
print(f"Dataset saved  -> model/student_data.csv  ({N} rows)")

# ── Train ──────────────────────────────────────────────────────────────────────
X = df.drop(columns=["marks"])
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"Model trained  ->  MAE: {mae:.2f}  |  R2: {r2:.4f}")

joblib.dump(model,  "model/marks_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Saved          -> model/marks_model.pkl  &  model/scaler.pkl")
