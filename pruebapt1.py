import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------------------------
# Functions

def randomcentroids_beginning(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assignation_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def centroids_update(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        datapoints = X[clusters == i]
        if len(datapoints) > 0:
            centroids[i] = datapoints.mean(axis=0)
        else:
            centroids[i] = X[np.random.randint(0, len(X))]
    return centroids

def kmeans(X, k, max_iters=100):
    centroids = randomcentroids_beginning(X, k)
    for _ in range(max_iters):
        clusters = assignation_clusters(X, centroids)
        centroids_new = centroids_update(X, clusters, k)
        if np.allclose(centroids, centroids_new, atol=1e-5):
            break
        centroids = centroids_new
    return centroids, clusters

def wcss_computation(X, centroids, clusters):
    distances = np.linalg.norm(X - centroids[clusters], axis=1)
    return np.sum(distances ** 2)

# ----------------------------------------------------------------------------------------------
# Main program

start_time = time.time()

df = pd.read_csv("proteins.csv")
df["sequence_length"] = df["sequence"].apply(len)
X = df[["enzyme", "hydrofob"]].values
K_range = range(1, 11)
wcss = []

for k in K_range:
    centroids, clusters = kmeans(X, k)
    wcss.append(wcss_computation(X, centroids, clusters))

k_optimal = np.argmin(np.diff(wcss)) + 2
centroids, clusters = kmeans(X, k_optimal)

df["cluster"] = clusters
cluster_means = df.groupby("cluster")["sequence_length"].mean()
largest_cluster = cluster_means.idxmax()
average_sequence_length = cluster_means[largest_cluster]
max_sequence_length = df[df["cluster"] == largest_cluster]["sequence_length"].max()
idx = df[df["cluster"] == largest_cluster]["sequence_length"].idxmax()
protein_id = df.loc[idx, "protid"]

end_time = time.time()
totaltime = end_time - start_time

# ----------------------------------------------------------------------------------------------
# Graphs

plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png')

plt.figure()
for i in range(k_optimal):
    points = X[clusters == i]
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Centroids', marker='X')
plt.xlabel("Enzyme")
plt.ylabel("Hydrofob")
plt.title("K-means Clustering with Centroids")
plt.legend()
plt.savefig('clusters_with_centroids.png')

plt.figure()
sns.heatmap(centroids, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Heatmap of Cluster Centroids")
plt.savefig('heatmap_centroids.png')

# ----------------------------------------------------------------------------------------------
# Results
print("\n------ RESULTS -----")
print(f"Total execution time: {totaltime:.2f} seconds")
print(f"Optimal number of clusters (k): {k_optimal}")
print(f"Cluster with the highest average sequence length: {largest_cluster}")
print(f"Average sequence length in this cluster: {average_sequence_length}")
print(f"Maximum sequence length in this cluster: {max_sequence_length}")
print(f"Protein ID with the maximum sequence length: {protein_id}")

