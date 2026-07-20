# Titanic Survival Prediction (SCT_ML_2)

A Logistic Regression model built on the Titanic dataset to predict passenger survival based on characteristics like gender, age, socio-economic status, and companions.

---

## 📊 Dataset & Features

The dataset is sourced from the historical Titanic passenger manifest. We drop highly missing or unstructured fields (`PassengerId`, `Name`, `Ticket`, `Cabin`) and focus on predictive features:

### Selected Features:
1. **Pclass**: Passenger Class (1st, 2nd, 3rd) - indicator of socio-economic status.
2. **Sex**: Passenger Gender.
3. **Age**: Passenger Age.
4. **SibSp**: Number of siblings/spouses aboard.
5. **Parch**: Number of parents/children aboard.
6. **Fare**: Passenger Fare.
7. **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## 🛠️ Data Preprocessing Pipeline

Using a `scikit-learn` `Pipeline` and `ColumnTransformer`, we implemented the following transformations:
- **Numerical Features** (`Age`, `Fare`, `SibSp`, `Parch`, `Pclass`):
  - Missing values imputed using the **median**.
  - Feature values scaled using standard normalization (`StandardScaler`).
- **Categorical Features** (`Sex`, `Embarked`):
  - Missing port entries imputed with the **most frequent port** ('S').
  - Encoded to dummy binary values using `OneHotEncoder(drop='first')` to prevent multicollinearity.

---

## 📈 Model Performance

After training a Logistic Regression model with an **80/20 train-test split**, the model achieved:

- **Accuracy:** `81.01%` (145/179 correct predictions)

### Confusion Matrix:
- **True Negatives (TN):** `90` (Survived = 0, predicted 0)
- **False Positives (FP):** `15` (Survived = 0, predicted 1)
- **False Negatives (FN):** `19` (Survived = 1, predicted 0)
- **True Positives (TP):** `55` (Survived = 1, predicted 1)

### Detailed Metrics:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Deceased)** | `0.83` | `0.86` | `0.84` | 105 |
| **1 (Survived)** | `0.79` | `0.74` | `0.76` | 74 |
| **Average / Total** | **`0.81`** | **`0.81`** | **`0.81`** | **179** |

### Feature Coefficients (Log-Odds Impact):
| Feature | Coefficient | Interpretation |
| :--- | :--- | :--- |
| **Fare** | `0.1287` | Slight positive impact (higher fares slightly increase survival probability). |
| **Parch** | `-0.0855` | Negligible negative impact. |
| **Embarked_Q** | `-0.1112` | Boarding at Queenstown has minor negative impact compared to Cherbourg (reference). |
| **SibSp** | `-0.3438` | Negative impact (having many siblings/spouses aboard reduces survival probability). |
| **Age** | `-0.3936` | Negative impact (older passengers were less likely to survive). |
| **Embarked_S** | `-0.4013` | Boarding at Southampton has negative impact compared to Cherbourg. |
| **Pclass** | `-0.7776` | Strong negative impact (lower class/higher number decrease survival probability). |
| **Sex_male** | `-2.5928` | Extremely strong negative impact (being male heavily decreased survival probability). |

---

## 🛠️ How to Run

1. Navigate to this directory:
   ```bash
   cd SCT_ML_2
   ```
2. Run the classification script:
   ```bash
   python titanic_classification.py
   ```
