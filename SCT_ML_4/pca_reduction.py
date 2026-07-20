import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# 1. Load and Standardize Data
# -----------------------------
print("Loading Iris dataset...")
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardization (Critical for PCA because it is scale-sensitive)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 2. Apply PCA (Reduce 4D to 2D)
# -----------------------------
print("Applying PCA to reduce dimensions from 4D to 2D...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance ratio
var_ratio = pca.explained_variance_ratio_
print(f"\nExplained Variance Ratio of Principal Components:")
print(f"Principal Component 1: {var_ratio[0]:.4f} ({var_ratio[0]*100:.2f}%)")
print(f"Principal Component 2: {var_ratio[1]:.4f} ({var_ratio[1]*100:.2f}%)")
print(f"Total Explained Variance (PC1 + PC2): {np.sum(var_ratio):.4f} ({np.sum(var_ratio)*100:.2f}%)")

# Create PCA DataFrame
pca_df = pd.DataFrame(
    data=X_pca,
    columns=['PC1', 'PC2']
)
pca_df['Target'] = y

# -----------------------------
# 3. Visualization
# -----------------------------
print("\nGenerating 2D PCA projection plot...")
plt.figure(figsize=(10, 8))

colors = ['#ff7675', '#74b9ff', '#55efc4']
for i, target_name in enumerate(target_names):
    sub_df = pca_df[pca_df['Target'] == i]
    plt.scatter(
        sub_df['PC1'],
        sub_df['PC2'],
        label=target_name.capitalize(),
        color=colors[i],
        s=70,
        alpha=0.85,
        edgecolors='w'
    )

plt.title("Iris Dataset - 2D PCA Projection", fontsize=16, fontweight='bold', pad=20)
plt.xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=12)
plt.ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=12)
plt.legend(title="Species", fontsize=11, title_fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save plot locally
dir_path = os.path.dirname(__file__)
plot_path = os.path.join(dir_path, "pca_projection.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"PCA Projection plot saved locally to: {plot_path}")
