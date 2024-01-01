import networkx as nx
import numpy as np
import pandas as pd
import save_results as sr


class KMeansCustom:
    def __init__(self, n_clusters, max_iters=1000):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            # Assign points to the nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # If centroids don't change much, stop iterating
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)


def spectral_partitioning_with_custom_kmeans(graph, num_partitions):
    adj_matrix = nx.adjacency_matrix(graph).todense()
    laplacian_matrix = nx.laplacian_matrix(graph).todense()#laplacian matrix contains information about the degree of nodes and the adjacency matrix of the original graph
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    fiedler_vector = eigenvectors[:, 1] # in accordance to the graph laplacian facts and courant-fischer minmax theorem the q1(eigenvector of the SECOND smallest eigenvalue)
    # is chosen. fiedler vector holds information about the algebraic connectivity of the graph, which can be further used in clustering algorithms to partition the graph

    fiedler_vector = fiedler_vector.reshape(-1, 1)

    # Create and fit KMeansCustom
    kmeans = KMeansCustom(n_clusters=num_partitions)
    kmeans.fit(fiedler_vector) # we then call the kmeans clustering algorithm on the fiedler vector to use it's components as dimensions
    # and cluster the nodes based on their algebraic connectivity.

    labels = kmeans.predict(fiedler_vector)

    partitions = []
    for i in range(num_partitions):
        partition = [node for node, label in zip(graph.nodes(), labels) if label == i]
        partitions.append(partition)

    return partitions


def calculate_edge_cuts_between_partitions(graph, partitions):
    num_partitions = len(partitions)
    edge_cuts = 0

    for u, v in graph.edges():
        u_partition = None
        v_partition = None

        for i in range(num_partitions):
            if u in partitions[i]:
                u_partition = i
            if v in partitions[i]:
                v_partition = i

        if u_partition is not None and v_partition is not None and u_partition != v_partition:
            edge_cuts += 1

    return edge_cuts


G = nx.erdos_renyi_graph(250, 0.3, seed=42)

# Perform spectral partitioning
num_partitions = 5
# partitions = spectral_partitioning(G, num_partitions)
customKmeansPartitions = spectral_partitioning_with_custom_kmeans(G, num_partitions)


for i in range(num_partitions):
    # print(f"Partition {i}: ", partitions[i])
    print(f"customKmeansPartition {i}: ", customKmeansPartitions[i])


# edge_cuts = calculate_edge_cuts_between_partitions(G, partitions)
# customKmeansEdgeCuts = calculate_edge_cuts_between_partitions(G, customKmeansPartitions)

# print("Total edge cuts between partitions:", edge_cuts)
# print("Total edge cuts between customKmeansPartitions:", customKmeansEdgeCuts)


def run_experiment_for_different_sizes(sizes, p, k):
    results = {"Graph Size": [], "Edge Cuts": []}
    for size in sizes:
        G_er = nx.erdos_renyi_graph(n=size, p=p, seed= 42)
        partitions = spectral_partitioning_with_custom_kmeans(G_er, k)
        edge_cuts = calculate_edge_cuts_between_partitions(G_er, partitions)
        results["Graph Size"].append(size)
        results["Edge Cuts"].append(edge_cuts)
    return pd.DataFrame(results)


# Run the experiment for different graph sizes

sizes = [100, 200, 300, 400, 500]
probability = 0.3
num_partitions = 5

experiment_results_SP = run_experiment_for_different_sizes(sizes, probability, num_partitions)
sr.save_results_to_csv(experiment_results_SP, "spectral_partitioning")


def run_experiment_for_different_partitions(size, p, partitions_list):
    results = {"Number of Partitions": [], "Edge Cuts": []}
    G_er = nx.erdos_renyi_graph(n=size, p=p)
    print(f"Running spectral partitioning for graph size {size} with different number of partitions:")
    for k in partitions_list:
        partitions = spectral_partitioning_with_custom_kmeans(G_er, k)
        edge_cuts = calculate_edge_cuts_between_partitions(G_er, partitions)
        results["Number of Partitions"].append(k)
        results["Edge Cuts"].append(edge_cuts)
        print(f"Number of partitions: {k}, Edge cuts: {edge_cuts}")
    return pd.DataFrame(results)


# Run the experiment for different partitions
size = 300
probability = 0.3
partitions_list = [2, 3, 4, 5]

experiment_results_SP = run_experiment_for_different_partitions(size, probability, partitions_list)

sr.save_results_to_csv(experiment_results_SP, "spectral_partitioning_different_partition")
