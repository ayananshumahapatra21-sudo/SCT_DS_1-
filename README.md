# Data Preprocessing & Categorical Encoding (SCT_DS_1-)

A Python-based data science pipeline demonstrating fundamental data preprocessing techniques, including missing value imputation and categorical encoding using `pandas` and `scikit-learn`.

---

## 🚀 Features

- 📊 **Basic Statistics:** Outputs concise dataset structural info (`df.info()`) and numerical/categorical distribution summaries (`df.describe()`).
- 🛠️ **Missing Value Imputation:**
  - **Mean Imputation:** Fills missing values in the `Age` column using the average of available ages.
  - **Median Imputation:** Fills missing values in the `Salary` column using the median salary to reduce the impact of potential outliers.
- 🏷️ **Categorical Encoding:**
  - **Label Encoding:** Encodes binary categorical data (`Gender` $\rightarrow$ `Gender_Label` as 0 or 1).
  - **One-Hot Encoding:** Encodes multi-class nominal data (`City` $\rightarrow$ binary dummy columns) without introducing ordinal bias.

---

## 📁 Project Structure

```text
data-preprocessing/
│
├── data.csv            # Input dataset with missing/categorical values
├── preprocess.py      # Main Python script for data processing
└── README.md           # Project documentation
```

---

## 📊 Dataset Structure

The input `data.csv` contains the following fields:

| Column | Type | Description |
| :--- | :--- | :--- |
| **Gender** | Categorical (Nominal) | Gender of the individual (e.g., Male, Female) |
| **Age** | Numerical (Continuous) | Age of the individual (contains missing values) |
| **Salary** | Numerical (Continuous) | Salary of the individual (contains missing values) |
| **City** | Categorical (Nominal) | City of residence (e.g., New York, Paris, Tokyo) |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayananshumahapatra21-sudo/SCT_DS_1-.git
   cd SCT_DS_1-
   ```

2. **Install dependencies:**
   Make sure you have `pandas` and `scikit-learn` installed:
   ```bash
   pip install pandas scikit-learn
   ```

3. **Run the script:**
   ```bash
   python preprocess.py
   ```
