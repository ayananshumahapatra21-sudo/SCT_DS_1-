# Heart Disease Prediction (SCT_ML_5)

A machine learning classification pipeline built on the Cleveland UCI Heart Disease dataset to predict whether a patient is likely to have heart disease based on various medical observations. We train and compare a **Logistic Regression** model and a **Decision Tree** model.

---

## 📊 Dataset Description

The dataset consists of 303 patients with 14 clinical features:
- **age**: Patient age (years)
- **sex**: Patient gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia blood disorder type (1-3)
- **target**: Diagnosis of heart disease (1 = disease, 0 = no disease)

---

## 📈 Model Performance & Comparison

After scaling continuous features and splitting data into **80% train and 20% test** sets (stratified by target variable), the models achieved the following metrics:

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **`80.33%`** | **`76.92%`** | **`90.91%`** | **`0.8333`** |
| **Decision Tree** | `75.41%` | `72.50%` | `87.88%` | `0.7945` |

*Note: In medical diagnosis, **Recall** is extremely important as we want to minimize False Negatives (predicting a patient is healthy when they actually have heart disease). Logistic Regression achieved a very strong recall of **90.91%**.*

---

## 🎨 Visualizations

The output plot **`heart_disease_evaluation.png`** contains:
1. **Confusion Matrices:**
   - Shows True Negatives, False Positives, False Negatives, and True Positives for both models side-by-side.
2. **ROC (Receiver Operating Characteristic) Curve:**
   - Plots True Positive Rate (TPR) vs. False Positive Rate (FPR) for both models.
   - Logistic Regression achieves a higher Area Under the Curve (AUC), showing superior overall diagnostic separation power compared to the Decision Tree.

---

## 🛠️ How to Run

1. Navigate to this directory:
   ```bash
   cd SCT_ML_5
   ```
2. Run the classification script:
   ```bash
   python heart_prediction.py
   ```
