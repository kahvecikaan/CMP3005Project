import networkx as nx
import numpy as np
from sklearn.cluster import KMeans  # For clustering

# Function to perform spectral partitioning
def spectral_partitioning(graph, num_partitions):
    adj_matrix = nx.adjacency_matrix(graph).todense()
    laplacian_matrix = nx.laplacian_matrix(graph).todense()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    fiedler_vector = eigenvectors[:, 1]

    kmeans = KMeans(n_clusters=num_partitions)
    kmeans.fit(fiedler_vector.reshape(-1, 1))
    labels = kmeans.labels_

    partitions = []
    for i in range(num_partitions):
        partition = [node for node, label in zip(graph.nodes(), labels) if label == i]
        partitions.append(partition)

    return partitions

def spectral_partitioning_without_kmeans(graph, num_partitions):
    adj_matrix = nx.adjacency_matrix(graph).todense()
    laplacian_matrix = nx.laplacian_matrix(graph).todense()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    fiedler_vector = eigenvectors[:, 1]

    # Sort nodes based on the Fiedler vector values
    sorted_nodes = [node for _, node in sorted(zip(fiedler_vector, graph.nodes()))]

    partition_size = len(sorted_nodes) // num_partitions
    partitions = [sorted_nodes[i * partition_size: (i + 1) * partition_size] for i in range(num_partitions - 1)]
    partitions.append(sorted_nodes[(num_partitions - 1) * partition_size:])

    return partitions


class KMeansCustom:
    def __init__(self, n_clusters, max_iters=300):
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


# Rest of your spectral partitioning function
def spectral_partitioning_with_custom_kmeans(graph, num_partitions):
    adj_matrix = nx.adjacency_matrix(graph).todense()
    laplacian_matrix = nx.laplacian_matrix(graph).todense()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    fiedler_vector = eigenvectors[:, 1]

    fiedler_vector = fiedler_vector.reshape(-1, 1)

    # Create and fit KMeansCustom
    kmeans = KMeansCustom(n_clusters=num_partitions)
    kmeans.fit(fiedler_vector)

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

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])

# Perform spectral partitioning
num_partitions = 2
partitions = spectral_partitioning(G, num_partitions)
noKmeansPartitions = spectral_partitioning_without_kmeans(G, num_partitions)
customKmeansPartitions = spectral_partitioning_with_custom_kmeans(G, num_partitions)


print(f"Nodes: {G.nodes()}")
print(f"Edges: {G.edges()}")
for i in range(num_partitions):
    print(f"Partition {i}: ", partitions[i])
    print(f"NoKmeansPartition {i}: ", noKmeansPartitions[i])
    print(f"customKmeansPartition {i}: ", customKmeansPartitions[i])


edge_cuts = calculate_edge_cuts_between_partitions(G, partitions)
noKmeansEdgeCuts = calculate_edge_cuts_between_partitions(G, noKmeansPartitions)
customKmeansEdgeCuts = calculate_edge_cuts_between_partitions(G, customKmeansPartitions)

print("Total edge cuts between partitions:", edge_cuts)
print("Total edge cuts between noKmeansPartitions:", noKmeansEdgeCuts)
print("Total edge cuts between customKmeansPartitions:", customKmeansEdgeCuts)