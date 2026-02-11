import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import threading

# THREADING VERSION (using threading.Thread)
#We need to: 
#1. Implement K-means algorithm for clustering using "enzyme" and "hydrofob" from the csv of proteins, using the euclidean distance and random centroids at the beginning
#2. Register the start and end time and print it
#3. Construct the elbow graph and plot it
#4. Find the optimal cluster number (k) and cluster the data using k
#5. Find the cluster with the highest length and compute its average sequence length
#6. Plot results: elbow graph, clusters with centroids, heat map using the values of clusters' centroids and for the cluster with highest sequence length, print id, highest value, average sequence length previously computed.

# ----------------------------------------------------------------------------------------------
# Functions

#Random centroids at the beginning
def randomcentroids_beginning(X, k):
    indices = random.sample(range(len(X)), k)
    return X[indices]

#Assign data to clusters
def assignation_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

#Update centroids after random initialization
def centroids_update(X, clusters, k):
    centroids = []
    for i in range(k):
        datapoints = X[clusters == i]
        if len(datapoints) > 0:
            centroids.append(np.mean(datapoints, axis=0))
        else:
            centroids.append(X[random.randint(0, len(X)-1)])
    return np.array(centroids)

# K-means algorithm without using the scikit learn library
def kmeans(X, k, max_iters=100):
    centroids = randomcentroids_beginning(X, k)
    for _ in range(max_iters):
        clusters = assignation_clusters(X, centroids)
        centroids_new = centroids_update(X, clusters, k)
        if np.allclose(centroids, centroids_new, atol=1e-5):
            break
        centroids = centroids_new
    return centroids, clusters

#Compute WCSS (within-cluster sum of squares)
def wcss_computation(X, centroids, clusters):
    distances = np.linalg.norm(X - centroids[clusters], axis=1)
    return np.sum(distances ** 2)

# Worker for each value of k
def worker(k, idx):
    centroids, clusters = kmeans(X, k)
    result = wcss_computation(X, centroids, clusters)
    with lock:
        wcss[idx] = result


# ----------------------------------------------------------------------------------------------
# Main program with threading

if __name__ == "__main__":
    start_time = time.time()

    df = pd.read_csv("proteins.csv")
    df["sequence_length"] = df["sequence"].apply(len)
    X = df[["enzyme", "hydrofob"]].values
    K_range = range(1, 11)

    wcss = [None] * len(K_range)
    threads = []
    lock = threading.Lock()



    # A thread per each k
    for idx, k in enumerate(K_range):
        t = threading.Thread(target=worker, args=(k, idx))
        threads.append(t)
        t.start()

    # Hold on for finishing all threads
    for t in threads:
        t.join()

    # Find the optimal k using the elbow method
    k_optimal = np.argmin(np.diff(wcss)) + 2
    centroids, clusters = kmeans(X, k_optimal)

    #Finding the cluster with the highest length and computing its average sequence length
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
    # Graphs & plots

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
    # Final print statements
    print("\n------ RESULTS -----")
    print(f"Total execution time: {totaltime:.2f} seconds")
    print(f"Optimal number of clusters (k): {k_optimal}")
    print(f"Cluster with the highest average sequence length: {largest_cluster}")
    print(f"Average sequence length in this cluster: {average_sequence_length}")
    print(f"Maximum sequence length in this cluster: {max_sequence_length}")
    print(f"Protein ID with the maximum sequence length: {protein_id}")