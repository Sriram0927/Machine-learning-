import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

# 1. LOAD DATA
data = pd.read_csv("seattle-weather.csv")

# 2. PREPROCESSING
data["RainToday"] = data["weather"].apply(lambda x: 1 if "rain" in x.lower() else 0)
data["RainTomorrow"] = data["RainToday"].shift(-1)
data = data.dropna()
data["temp_range"] = data["temp_max"] - data["temp_min"]
features = ["precipitation", "temp_max", "temp_min", "temp_range", "wind"]
X = data[features].fillna(data[features].mean())
y = data["RainTomorrow"].astype(int)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)
# 3. MODEL BUILD & TRAIN (Random Forest)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. MODEL EVALUATION
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
importances = model.feature_importances_
plt.figure(figsize=(6,4))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")
plt.show()
print("\n")

# 5. SAVE / LOAD MODEL
joblib.dump(model, "rain_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully.")
loaded_model = joblib.load("rain_rf_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
# 6. NEW DATA PREDICTION
new_data = pd.DataFrame({
    "precipitation": [0.5],
    "temp_max": [10],
    "temp_min": [6],
    "temp_range": [4],
    "wind": [3.5]
})
new_scaled = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_scaled)
print("Rain Prediction for New Data (1 = Rain, 0 = No Rain):", int(prediction[0]))
