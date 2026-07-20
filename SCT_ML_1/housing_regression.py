import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load and Save Dataset
# -----------------------------
print("Fetching California Housing dataset...")
california = fetch_california_housing(as_frame=True)
df = california.frame

# Rename Target to Price
df = df.rename(columns={'MedHouseVal': 'House_Price'})

# Save dataset locally as housing.csv
csv_path = os.path.join(os.path.dirname(__file__), "housing.csv")
df.to_csv(csv_path, index=False)
print(f"Dataset saved locally to: {csv_path}")

print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X = df.drop(columns=['House_Price'])
y = df['House_Price']

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Train Linear Regression Model
# -----------------------------
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# 5. Model Evaluation
# -----------------------------
# Predict on test set
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Print feature coefficients
print("\nModel Coefficients:")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print(coefficients)

print(f"\nIntercept: {model.intercept_:.4f}")
