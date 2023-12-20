import networkx as nx
import random

def add_default_weights(graph, default_weight=1):
    for u, v in graph.edges():
        if 'weight' not in graph[u][v]:
            graph[u][v]['weight'] = default_weight

def coarsen_graph(graph):
    add_default_weights(graph)  # Ensure the graph is weighted

    coarse_graph = graph.copy()
    merged = set()

    while len(coarse_graph.nodes()) > 1:
        nodes = list(coarse_graph.nodes())
        random.shuffle(nodes)

        for node in nodes:
            if node in merged:
                continue

            # Find the heaviest edge for the current node
            max_weight = -1
            max_neighbor = None
            for neighbor in coarse_graph.neighbors(node):
                weight = coarse_graph[node][neighbor]['weight']
                if weight > max_weight and neighbor not in merged:
                    max_weight = weight
                    max_neighbor = neighbor

            # Merge the nodes if a heavy-edge neighbor is found
            if max_neighbor:
                for neighbor in list(coarse_graph.neighbors(max_neighbor)):
                    if neighbor != node:
                        weight1 = coarse_graph[max_neighbor][neighbor]['weight']
                        if coarse_graph.has_edge(node, neighbor):
                            weight2 = coarse_graph[node][neighbor]['weight']
                            coarse_graph[node][neighbor]['weight'] = weight1 + weight2
                        else:
                            coarse_graph.add_edge(node, neighbor, weight=weight1)
                coarse_graph.remove_node(max_neighbor)
                merged.add(node)
                merged.add(max_neighbor)

        if not max_neighbor:  # If no merges happened in this iteration, break
            break

    return coarse_graph


def calculate_edge_cut_increase(graph, node, partition):
    """
    Calculate the increase in edge cut if 'node' is added to 'partition'.
    """
    increase = 0
    for neighbor in graph.neighbors(node):
        if neighbor not in partition:
            increase += 1
    return increase


def partition_coarsened_graph(coarsened_graph, k):
    partitions = {i: set() for i in range(k)}
    node_list = list(coarsened_graph.nodes())
    node_list.sort(key=lambda x: coarsened_graph.degree(x), reverse=True)

    # Average partition size for balancing
    avg_partition_size = len(node_list) / k

    for node in node_list:
        best_partition = None
        min_edge_cut_increase = float('inf')
        min_size_difference = float('inf')

        for i in range(k):
            current_partition_size = len(partitions[i])

            # Prioritize filling empty partitions first
            if current_partition_size == 0:
                best_partition = i
                break

            edge_cut_increase = calculate_edge_cut_increase(coarsened_graph, node, partitions[i])
            size_difference = abs(current_partition_size + 1 - avg_partition_size)

            # Choose partition that balances edge-cut increase and size difference
            if edge_cut_increase < min_edge_cut_increase or (
                    edge_cut_increase == min_edge_cut_increase and size_difference < min_size_difference):
                best_partition = i
                min_edge_cut_increase = edge_cut_increase
                min_size_difference = size_difference

        partitions[best_partition].add(node)

    return partitions


# Example usage
G = nx.erdos_renyi_graph(10, 0.3, seed=42)
print("Nodes before coarsening:", G.nodes())
coarse_G = coarsen_graph(G)
print("Nodes after coarsening:", coarse_G.nodes())

k = 3
partitions = partition_coarsened_graph(coarse_G, k)
print("Partitions:", partitions)