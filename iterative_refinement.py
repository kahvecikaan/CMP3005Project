import networkx as nx
import random

def initial_partition(graph, k):
    """
    Improved initial partitioning.
    Assigns nodes to partitions in a round-robin fashion for better balance.
    """
    partitions = {i: set() for i in range(k)}
    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Shuffle nodes for randomness

    for i, node in enumerate(nodes):
        partition = i % k
        partitions[partition].add(node)
    return partitions


def calculate_edge_cut(graph, partitions):
    """
    Calculates the number of edges that cross between different partitions.
    """
    edge_cut = 0
    for edge in graph.edges():
        for part in partitions.values():
            if edge[0] in part and edge[1] not in part:
                edge_cut += 1
    return edge_cut


def refine_partitions(graph, partitions):
    """
    Refinement algorithm to move nodes between partitions
    to reduce the edge-cut while maintaining balance.
    """
    improved = True
    while improved:
        improved = False
        for part_id, part in partitions.items():
            for node in list(part):
                best_move = None
                best_edge_cut = calculate_edge_cut(graph, partitions)
                for other_part_id, other_part in partitions.items():
                    if part_id != other_part_id:
                        # Move node only if it balances the partition sizes
                        if len(part) > len(other_part):
                            part.remove(node)
                            other_part.add(node)
                            new_edge_cut = calculate_edge_cut(graph, partitions)
                            if new_edge_cut < best_edge_cut:
                                best_move = other_part_id
                                best_edge_cut = new_edge_cut
                            other_part.remove(node)
                            part.add(node)
                if best_move is not None:
                    partitions[best_move].add(node)
                    part.remove(node)
                    improved = True
    return partitions


# Example usage
G = nx.Graph()
G.add_edges_from([(0, 4), (4, 3), (4, 6), (1, 2), (2, 3), (2, 5)])  # Custom graph

print("Nodes:", G.nodes())
print("Edges:", G.edges())

k = 5  # Number of partitions
partitions = initial_partition(G, k)
partitions = refine_partitions(G, partitions)

print("Partitions:", partitions)
print("Edge cut:", calculate_edge_cut(G, partitions))


G_er = nx.erdos_renyi_graph(n=21, p=0.5)
print("Nodes:", G_er.nodes())
print("Edges:", G_er.edges())
k2 = 13
partitions = initial_partition(G_er, k2)
partitions = refine_partitions(G_er, partitions)
print("Partitions:", partitions)
print("Edge cut:", calculate_edge_cut(G_er, partitions))
