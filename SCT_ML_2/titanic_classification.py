import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Download and Save Dataset
# -----------------------------
csv_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
csv_path = os.path.join(os.path.dirname(__file__), "titanic.csv")

if not os.path.exists(csv_path):
    print("Downloading Titanic dataset from GitHub...")
    urllib.request.urlretrieve(csv_url, csv_path)
    print("Dataset saved locally.")
else:
    print("Titanic dataset already exists locally.")

df = pd.read_csv(csv_path)
print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# 2. Feature Selection & Target Separation
# -----------------------------
# We drop PassengerId, Name, Ticket, Cabin as they are identifiers, free text, or highly missing.
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
y = df['Survived']

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Preprocessing Pipeline
# -----------------------------
# Numerical columns: Age, Fare, SibSp, Parch. Impute missing Age with median, scale features.
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical columns: Sex, Embarked. Impute missing Embarked with mode, encode.
categorical_features = ['Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False)) # drop='first' avoids multicollinearity
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 5. Train Logistic Regression Model
# -----------------------------
# Create the machine learning pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

print("\nTraining Logistic Regression model...")
clf.fit(X_train, y_train)

# -----------------------------
# 6. Model Evaluation
# -----------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Get feature names after one-hot encoding
cat_encoder = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

coefficients = clf.named_steps['classifier'].coef_[0]

print("\nModel Coefficients (Impact on Survival Probability):")
coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)
print(coeff_df)
