import networkx as nx
from collections import deque


def enhanced_dfs_partition(graph):
    """ Perform DFS with a focus on creating cohesive partitions. """
    visited = set()
    partitions = []
    for start_node in sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True):
        if start_node not in visited:
            stack = deque([start_node])
            path = set()
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    path.add(current)
                    # Prioritize unvisited neighbors with higher degrees
                    for neighbor in sorted(graph[current], key=lambda n: graph.degree(n), reverse=True):
                        if neighbor not in visited:
                            stack.append(neighbor)
            partitions.append(path)
    return partitions


def adjust_partitions(partitions, k, graph):
    """ Adjust partitions to match the desired count while minimizing edge cuts. """
    while len(partitions) != k:
        if len(partitions) < k:
            # Split the largest partition
            largest_partition = max(partitions, key=len)
            partitions.remove(largest_partition)
            half = len(largest_partition) // 2
            partitions.extend([set(list(largest_partition)[:half]), set(list(largest_partition)[half:])])
        else:
            # Merge partitions with the least edge cuts between them
            min_edge_cuts = float('inf')
            pair_to_merge = None
            for i in range(len(partitions)):
                for j in range(i + 1, len(partitions)):
                    edge_cuts = sum(1 for u in partitions[i] for v in partitions[j] if graph.has_edge(u, v))
                    if edge_cuts < min_edge_cuts:
                        min_edge_cuts = edge_cuts
                        pair_to_merge = (i, j)
            partitions[pair_to_merge[0]].update(partitions[pair_to_merge[1]])
            partitions.pop(pair_to_merge[1])
    return partitions


def calculate_edge_cuts(graph, partitions):
    """ Calculate the number of edge cuts for the given partitions. """
    return sum(1 for u, v in graph.edges() if any(u in p and v not in p for p in partitions))


# Example Usage
G = nx.erdos_renyi_graph(500, 0.3)
initial_partitions = enhanced_dfs_partition(G)
k = 15  # Desired number of partitions
final_partitions = adjust_partitions(initial_partitions, k, G)
edge_cuts = calculate_edge_cuts(G, final_partitions)

print(f"Number of Partitions: {len(final_partitions)}")
print(f"Edge cuts: {edge_cuts}")
