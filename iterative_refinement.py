import networkx as nx
import random
import pandas as pd
import save_results as sr


def initial_partition(graph, k):
    """
    Assigns nodes to partitions in a round-robin fashion for better balance.
    """
    partitions = {i: set() for i in range(k)}
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    for i, node in enumerate(nodes):
        partition = i % k
        partitions[partition].add(node)
    return partitions


def calculate_edge_cut_incremental(graph, partitions, node, old_part, new_part):
    """
    Calculates the change in edge cut when moving a node from one partition to another.
    """
    change = 0
    for neighbor in graph.neighbors(node):
        if neighbor in partitions[old_part] and neighbor not in partitions[new_part]:
            change -= 1  # Edge cut reduced
        elif neighbor not in partitions[old_part] and neighbor in partitions[new_part]:
            change += 1  # Edge cut increased
    return change


def refine_partitions(graph, partitions, k):
    """
    Refines the partitioning by iteratively moving nodes to reduce edge cuts.
    """
    improved = True
    while improved:
        improved = False
        for part_id, part in partitions.items():
            for node in list(part):
                best_change = 0
                best_new_part = None
                for new_part_id, new_part in partitions.items():
                    if part_id != new_part_id and len(new_part) < len(part):
                        change = calculate_edge_cut_incremental(graph, partitions, node, part_id, new_part_id)
                        if change < best_change:
                            best_change = change
                            best_new_part = new_part_id
                if best_new_part is not None:
                    partitions[best_new_part].add(node)
                    part.remove(node)
                    improved = True
                    break  # Break to re-evaluate after each move
            if improved:
                break  # Break the outer loop if a move was made

    # Ensure all partitions have approximately equal size
    while any(len(part) > len(graph.nodes()) // k + 1 for part in partitions.values()):
        # Move nodes from larger to smaller partitions
        larger_part = max(partitions, key=lambda x: len(partitions[x]))
        smaller_part = min(partitions, key=lambda x: len(partitions[x]))
        node_to_move = next(iter(partitions[larger_part]))
        partitions[larger_part].remove(node_to_move)
        partitions[smaller_part].add(node_to_move)

    return partitions


def calculate_total_edge_cut(graph, partitions):
    """
    Calculates the total number of edge cuts in the current partitioning.
    """
    edge_cut = 0
    for u, v in graph.edges():
        for part in partitions.values():
            if u in part and v not in part:
                edge_cut += 1
    return edge_cut


def run_experiment_for_different_sizes(sizes, p, k):
    results = {"Graph Size": [], "Edge Cuts": []}
    for size in sizes:
        G_er = nx.erdos_renyi_graph(n=size, p=p)
        partitions = initial_partition(G_er, k)
        refined_partitions = refine_partitions(G_er, partitions, k)
        edge_cut = calculate_total_edge_cut(G_er, refined_partitions)
        results["Graph Size"].append(size)
        results["Edge Cuts"].append(edge_cut)
    return pd.DataFrame(results)


def run_experiment_for_different_partitions(size, p, partitions_list):
    results = {"Number of Partitions": [], "Edge Cuts": []}
    G_er = nx.erdos_renyi_graph(n=size, p=p)
    for k in partitions_list:
        initial_parts = initial_partition(G_er, k)
        refined_partitions = refine_partitions(G_er, initial_parts, k)
        edge_cut = calculate_total_edge_cut(G_er, refined_partitions)
        results["Number of Partitions"].append(k)
        results["Edge Cuts"].append(edge_cut)
    return pd.DataFrame(results)


# Example usage
G_er = nx.erdos_renyi_graph(n=500, p=0.3)
k = 5
partitions = initial_partition(G_er, k)
refined_partitions = refine_partitions(G_er, partitions, k)
edge_cut = calculate_total_edge_cut(G_er, refined_partitions)

print(f"Partitions: {refined_partitions}")
print(f"Edge cut: {edge_cut}")


# Run the experiment for different graph sizes
graph_sizes = [50, 100, 150, 200, 250]
probability = 0.3
num_partitions = 5

experiment_results = run_experiment_for_different_sizes(graph_sizes, probability, num_partitions)

# save results to csv
sr.save_results_to_csv(experiment_results, "iterative_refinement")


# Run the experiment for different number of partitions
size = 300
probability = 0.3
partitions_list = [2, 3, 4, 5]

experiment_results_iterative = run_experiment_for_different_partitions(size, probability, partitions_list)

# save results to csv
sr.save_results_to_csv(experiment_results_iterative, "iterative_refinement_different_partition")
