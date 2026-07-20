# Iris K-Means Clustering (SCT_ML_3)

An unsupervised learning model implementing K-Means Clustering on the classic Iris Flower dataset. The optimal number of clusters ($K$) is determined using the Elbow Method and visualized in 2D space.

---

## 📊 Dataset Description

The Iris dataset contains 150 instances of iris flowers belonging to three species: Setosa, Versicolor, and Virginica. The features included are:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 📈 Methodology & Results

### 1. The Elbow Method (Finding Optimal K)
We ran the clustering algorithm with values of $K$ from 1 to 10 and plotted the **Within-Cluster Sum of Squares (WCSS)**. 
- A distinct "elbow" point is observed at **$K=3$**. This matches the actual number of biological species in the Iris dataset (Setosa, Versicolor, Virginica), validating our choice.

### 2. Clustering Performance
Training K-Means with $K=3$ produces the following cluster distribution:
- **Cluster 0:** 62 samples
- **Cluster 1:** 50 samples
- **Cluster 2:** 38 samples

---

## 🎨 Visualizations

The generated plot **`iris_clusters.png`** contains two side-by-side figures:
1. **Elbow Curve:** Shows the WCSS decrease across values of $K$, marking $K=3$ as the optimal elbow point.
2. **2D Scatter Plot (Petal Length vs. Petal Width):** Displays the clusters assigned by K-Means along with their centroids (marked with a black `X`).

---

## 🛠️ How to Run

1. Navigate to this directory:
   ```bash
   cd SCT_ML_3
   ```
2. Run the script to perform clustering and regenerate plots:
   ```bash
   python kmeans_iris.py
   ```
