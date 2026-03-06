import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
dataset = pd.read_csv("Mall_Customers.csv")
print(dataset.head())

# -------------------------------
# Step 2: Select Features
# Basis: Income vs Spending
# -------------------------------
X = dataset.iloc[:, [3, 4]].values

# -------------------------------
# Step 3: Elbow Method
# -------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# -------------------------------
# Step 4: Train Model
# -------------------------------
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# -------------------------------
# Step 5: Silhouette Score
# -------------------------------
score = silhouette_score(X, y_kmeans)
print("Silhouette Score :", score)

# -------------------------------
# Step 6: Cluster Names
# -------------------------------
cluster_names = {
    0: "Premium Customers",
    1: "Careful Customers",
    2: "Impulsive Customers",
    3: "Budget Customers",
    4: "Standard Customers"
}

# Add cluster column
dataset['Cluster'] = y_kmeans
dataset['Cluster Name'] = dataset['Cluster'].map(cluster_names)

print(dataset.head())

# -------------------------------
# Step 7: Visualization
# -------------------------------
plt.figure(figsize=(10,7))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, label=cluster_names[0])
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, label=cluster_names[1])
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, label=cluster_names[2])
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, label=cluster_names[3])
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, label=cluster_names[4])

# Centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300,
            label='Centroids')

plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()
