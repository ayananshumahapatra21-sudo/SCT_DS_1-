import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# -----------------------------
# 1. Load and Save Dataset
# -----------------------------
print("Loading Iris dataset...")
iris = load_iris(as_frame=True)
df = iris.frame

# Rename target column to Species and map integer class names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['Species'] = df['target'].map(species_map)
df_clean = df.drop(columns=['target'])

# Save raw dataset locally as iris.csv
dir_path = os.path.dirname(__file__)
csv_path = os.path.join(dir_path, "iris.csv")
df_clean.to_csv(csv_path, index=False)
print(f"Dataset saved locally to: {csv_path}")

# Features for clustering
X = df_clean.drop(columns=['Species'])

# -----------------------------
# 2. Elbow Method to Find Optimal K
# -----------------------------
print("\nRunning Elbow Method to find optimal number of clusters...")
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# -----------------------------
# 3. Train K-Means on Optimal K (K=3)
# -----------------------------
print("Training K-Means model with K=3...")
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X)

print("\nCluster Value Counts:")
print(df_clean['Cluster'].value_counts())

# -----------------------------
# 4. Visualization & Plotting
# -----------------------------
print("\nGenerating visualization plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Elbow Method
ax1.plot(k_range, wcss, marker='o', linestyle='--', color='#6c5ce7', linewidth=2, markersize=8)
ax1.set_title("Elbow Method for Optimal K", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.axvline(x=3, color='#e74c3c', linestyle=':', label='Elbow Point (K=3)', linewidth=2)
ax1.legend(fontsize=11)

# Subplot 2: Cluster Visualization (Petal Length vs Petal Width)
colors = ['#ff7675', '#74b9ff', '#55efc4']
for cluster_id in range(optimal_k):
    cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
    ax2.scatter(
        cluster_data['petal length (cm)'],
        cluster_data['petal width (cm)'],
        label=f'Cluster {cluster_id}',
        color=colors[cluster_id],
        s=60,
        alpha=0.85,
        edgecolors='w'
    )

# Plot cluster centroids
centroids = kmeans.cluster_centers_
# Features in centroids: sepal_len, sepal_wid, petal_len, petal_wid
# We want petal_len (index 2) and petal_wid (index 3)
ax2.scatter(
    centroids[:, 2],
    centroids[:, 3],
    s=250,
    c='#2d3436',
    marker='X',
    label='Centroids',
    edgecolors='w',
    linewidth=2
)

ax2.set_title("Iris Clusters (Petal Length vs Petal Width)", fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel("Petal Length (cm)", fontsize=12)
ax2.set_ylabel("Petal Width (cm)", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save the plot locally
plot_path = os.path.join(dir_path, "iris_clusters.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Visualizations saved locally to: {plot_path}")
