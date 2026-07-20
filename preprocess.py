import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load dataset
df = pd.read_csv("data.csv")

# -----------------------------
# 1. Basic Statistics
# -----------------------------
print("Dataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include='all'))

# -----------------------------
# 2. Handle Missing Values
# -----------------------------

# Mean Imputation (for numerical columns)
mean_imputer = SimpleImputer(strategy='mean')
df[['Age']] = mean_imputer.fit_transform(df[['Age']])

# Median Imputation
median_imputer = SimpleImputer(strategy='median')
df[['Salary']] = median_imputer.fit_transform(df[['Salary']])

print("\nDataset after handling missing values:")
print(df)

# -----------------------------
# 3. Encoding Categorical Data
# -----------------------------

# Label Encoding
le = LabelEncoder()
df['Gender_Label'] = le.fit_transform(df['Gender'])

print("\nAfter Label Encoding:")
print(df[['Gender', 'Gender_Label']])

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)

encoded = ohe.fit_transform(df[['City']])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(['City'])
)

# Merge with original dataset
df = pd.concat([df, encoded_df], axis=1)

print("\nAfter One-Hot Encoding:")
print(df)
