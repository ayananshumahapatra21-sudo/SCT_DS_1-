import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# -----------------------------
# 1. Download and Save Dataset
# -----------------------------
csv_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
dir_path = os.path.dirname(__file__)
csv_path = os.path.join(dir_path, "heart.csv")

if not os.path.exists(csv_path):
    print("Downloading Heart Disease UCI dataset...")
    urllib.request.urlretrieve(csv_url, csv_path)
    print("Dataset saved locally.")
else:
    print("Heart Disease dataset already exists locally.")

df = pd.read_csv(csv_path)
print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values Count:")
print(df.isnull().sum())

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# 3. Preprocessing (Feature Scaling)
# -----------------------------
# Continuous numerical features that require scaling
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Categorical features that are already encoded as integers
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', 'passthrough', categorical_features)
    ]
)

# -----------------------------
# 4. Train Models
# -----------------------------
# Define Pipelines
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
])

print("\nTraining Logistic Regression model...")
lr_pipeline.fit(X_train, y_train)

print("Training Decision Tree model...")
dt_pipeline.fit(X_train, y_train)

# -----------------------------
# 5. Model Evaluation
# -----------------------------
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    return metrics

lr_metrics = evaluate_model(lr_pipeline, X_test, y_test)
dt_metrics = evaluate_model(dt_pipeline, X_test, y_test)

print("\n==========================================")
print("Model Comparison Results:")
print("==========================================")
print(f"Logistic Regression:")
print(f"  Accuracy : {lr_metrics['accuracy']:.4f}")
print(f"  Precision: {lr_metrics['precision']:.4f}")
print(f"  Recall   : {lr_metrics['recall']:.4f}")
print(f"  F1 Score : {lr_metrics['f1']:.4f}")

print(f"\nDecision Tree:")
print(f"  Accuracy : {dt_metrics['accuracy']:.4f}")
print(f"  Precision: {dt_metrics['precision']:.4f}")
print(f"  Recall   : {dt_metrics['recall']:.4f}")
print(f"  F1 Score : {dt_metrics['f1']:.4f}")

# -----------------------------
# 6. Plot Confusion Matrices and ROC Curves
# -----------------------------
print("\nGenerating evaluation visualizations...")
fig = plt.figure(figsize=(15, 12))

# Subplot 1: Logistic Regression Confusion Matrix
ax1 = plt.subplot(2, 2, 1)
lr_cm = confusion_matrix(y_test, lr_metrics['y_pred'])
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1, annot_kws={"size": 14})
ax1.set_title("Logistic Regression Confusion Matrix", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("Predicted Label", fontsize=11)
ax1.set_ylabel("True Label", fontsize=11)
ax1.set_xticklabels(['No Disease', 'Disease'])
ax1.set_yticklabels(['No Disease', 'Disease'])

# Subplot 2: Decision Tree Confusion Matrix
ax2 = plt.subplot(2, 2, 2)
dt_cm = confusion_matrix(y_test, dt_metrics['y_pred'])
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax2, annot_kws={"size": 14})
ax2.set_title("Decision Tree Confusion Matrix", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel("Predicted Label", fontsize=11)
ax2.set_ylabel("True Label", fontsize=11)
ax2.set_xticklabels(['No Disease', 'Disease'])
ax2.set_yticklabels(['No Disease', 'Disease'])

# Subplot 3: ROC Curve Comparison
ax3 = plt.subplot(2, 1, 2)

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_metrics['y_prob'])
roc_auc_lr = auc(fpr_lr, tpr_lr)
ax3.plot(fpr_lr, tpr_lr, color='#0984e3', lw=3, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')

# Decision Tree ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_metrics['y_prob'])
roc_auc_dt = auc(fpr_dt, tpr_dt)
ax3.plot(fpr_dt, tpr_dt, color='#e17055', lw=3, label=f'Decision Tree (AUC = {roc_auc_dt:.4f})')

# Diagonal baseline
ax3.plot([0, 1], [0, 1], color='#b2bec3', lw=2, linestyle='--')

ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax3.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax3.set_title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold', pad=15)
ax3.legend(loc="lower right", fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save visual report
plot_path = os.path.join(dir_path, "heart_disease_evaluation.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Visual evaluations saved locally to: {plot_path}")
