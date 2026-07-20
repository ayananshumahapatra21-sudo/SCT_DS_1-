# Iris Dimensionality Reduction via PCA (SCT_ML_4)

An unsupervised learning project applying Principal Component Analysis (PCA) to reduce the dimensionality of the Iris Flower dataset from 4 dimensions down to 2 dimensions for visual analysis.

---

## 🔍 Understanding PCA

Principal Component Analysis (PCA) is a linear dimensionality reduction technique that finds orthogonal directions (principal components) of maximum variance in high-dimensional data.
- It transforms the features into a new coordinate system where the first principal component (PC1) accounts for the largest possible variance, the second (PC2) accounts for the next largest, and so on.

---

## 📈 Analysis & Results

### 1. Data Normalization
Features were standardized using a `StandardScaler` to have a mean of 0 and variance of 1. Normalization is critical before applying PCA, as features with larger scales would dominate the variance calculation otherwise.

### 2. Explained Variance Analysis
By reducing the Iris dataset from 4 features down to 2 components, we achieved the following explained variance ratios:
- **Principal Component 1 (PC1):** `72.96%` of the total dataset variance.
- **Principal Component 2 (PC2):** `22.85%` of the total dataset variance.
- **Total Information Preserved (PC1 + PC2):** **`95.81%`** of the total variance.

*This means we reduced the feature space complexity by 50% (4D to 2D) while preserving over 95% of the information!*

---

## 🎨 Visualizations

The generated scatter plot **`pca_projection.png`** shows the 2D projection:
- **PC1** is on the X-axis and **PC2** is on the Y-axis.
- The data points are colored by their actual Iris flower class (**Setosa**, **Versicolor**, **Virginica**).
- **Setosa** forms a completely distinct and linearly separable cluster along the PC1 axis, while **Versicolor** and **Virginica** show slight overlap but remain highly distinguishable.

---

## 🛠️ How to Run

1. Navigate to this directory:
   ```bash
   cd SCT_ML_4
   ```
2. Run the script to execute PCA and regenerate plots:
   ```bash
   python pca_reduction.py
   ```
