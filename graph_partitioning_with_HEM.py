import random
import networkx as nx
import pandas as pd
import save_results as sr


# Assign default weights to unweighted graph edges
def assign_default_weights(G):
    for u, v in G.edges():
        G[u][v]['weight'] = 1


# Perform Heavy Edge Matching (HEM)
def heavy_edge_matching(g):
    matched = set()
    matchings = []

    for u, v, data in sorted(g.edges(data=True), key=lambda x: x[2]['weight'], reverse=True):
        if u not in matched and v not in matched:
            matchings.append((u, v))
            matched.update([u, v])

    return matchings


# Coarsen the graph using the matchings from HEM
def coarsen_graph(G, matchings):
    coarse_G = nx.Graph()
    mapping = {}

    # First, handle matched nodes
    for u, v in matchings:
        coarse_node = len(coarse_G.nodes())
        coarse_G.add_node(coarse_node)
        mapping[u] = coarse_node
        mapping[v] = coarse_node

    # Add unmatched nodes as individual entries
    unmatched_nodes = set(G.nodes()) - set(mapping.keys())
    for node in unmatched_nodes:
        coarse_node = len(coarse_G.nodes())
        coarse_G.add_node(coarse_node)
        mapping[node] = coarse_node

    # Add edges to the coarse graph
    for u, v, data in G.edges(data=True):
        new_u, new_v = mapping[u], mapping[v]
        if coarse_G.has_edge(new_u, new_v):
            coarse_G[new_u][new_v]['weight'] += data['weight']
        else:
            coarse_G.add_edge(new_u, new_v, weight=data['weight'])

    return coarse_G, mapping


def partition_graph_balanced(G, num_partitions):
    partitions = {i: set() for i in range(num_partitions)}
    nodes = list(G.nodes())
    random.shuffle(nodes)

    for node in nodes:
        # Assign to the least populated partition
        min_part = min(partitions, key=lambda k: len(partitions[k]))
        partitions[min_part].add(node)
    return partitions


# Improved refinement that balances partitions and minimizes edge cuts
def refine_partitions_balanced(G, partitions, num_iterations=10):
    for _ in range(num_iterations):
        for part_id, nodes in list(partitions.items()):
            for node in list(nodes):
                current_cut = calculate_edge_cuts(G, partitions)
                best_move = (None, current_cut)
                for other_part_id, other_nodes in partitions.items():
                    if other_part_id != part_id and len(partitions[other_part_id]) < len(nodes):
                        # Move node and check edge cuts
                        partitions[part_id].remove(node)
                        partitions[other_part_id].add(node)
                        new_cut = calculate_edge_cuts(G, partitions)
                        if new_cut < best_move[1]:
                            best_move = (other_part_id, new_cut)
                        # Revert the move
                        partitions[other_part_id].remove(node)
                        partitions[part_id].add(node)
                # Make the best move
                if best_move[0] is not None:
                    partitions[part_id].remove(node)
                    partitions[best_move[0]].add(node)
    return partitions


# Uncoarsen and refine the partitions
def uncoarsen_and_refine(partitions, mapping):
    # Create an inverse mapping that maps each node in the coarse graph
    # back to the original nodes in G
    inverse_mapping = {}
    for orig_node, coarse_node in mapping.items():
        inverse_mapping.setdefault(coarse_node, []).append(orig_node)

    # Initialize the original partitions
    original_partitions = {i: set() for i in partitions}

    # Translate partitions back to the original graph using the inverse mapping
    for part_id, coarse_nodes in partitions.items():
        for coarse_node in coarse_nodes:
            original_nodes = inverse_mapping[coarse_node]
            original_partitions[part_id].update(original_nodes)

    return original_partitions


# Calculate the number of edge cuts in the partitions
def calculate_edge_cuts(G, partitions):
    edge_cuts = 0
    seen_edges = set()

    for u, v in G.edges():
        if (u, v) not in seen_edges and (v, u) not in seen_edges:
            for part in partitions.values():
                if (u in part and v not in part) or (v in part and u not in part):
                    edge_cuts += 1
                    seen_edges.add((u, v))
                    break

    return edge_cuts


# Combine all the functions into the multiway partitioning algorithm
def multiway_partition(G, num_partitions):
    assign_default_weights(G)
    matchings = heavy_edge_matching(G)
    coarse_G, mapping = coarsen_graph(G, matchings)
    partitions = partition_graph_balanced(coarse_G, num_partitions)
    original_partitions = uncoarsen_and_refine(partitions, mapping)  # Corrected line
    refined_partitions = refine_partitions_balanced(G, original_partitions)
    edge_cuts = calculate_edge_cuts(G, refined_partitions)
    return refined_partitions, edge_cuts


# Main Execution
# g = nx.erdos_renyi_graph(513, 0.3, 23)
# num_partitions = 5  # Define the number of partitions
# final_partitions, total_edge_cuts = multiway_partition(g, num_partitions)

# print("Final Partitions:", final_partitions)
# print("Total Edge Cuts:", total_edge_cuts)


def run_experiment_for_different_sizes(sizes, p, k):
    results = {"Graph Size": [], "Edge Cuts": []}
    for size in sizes:
        G_er = nx.erdos_renyi_graph(n=size, p=p)
        _, edge_cuts = multiway_partition(G_er, k)
        results["Graph Size"].append(size)
        results["Edge Cuts"].append(edge_cuts)
    return pd.DataFrame(results)


def run_experiment_for_different_partitions(size, p, partitions_list):
    results = {"Number of Partitions": [], "Edge Cuts": []}
    G_er = nx.erdos_renyi_graph(n=size, p=p)
    for k in partitions_list:
        _, edge_cuts = multiway_partition(G_er, k)
        results["Number of Partitions"].append(k)
        results["Edge Cuts"].append(edge_cuts)
    return pd.DataFrame(results)


# Run the experiment for different graph sizes
sizes = [50, 100, 150, 200, 250]
probability = 0.3
num_partitions = 5

experiment_results_HEM = run_experiment_for_different_sizes(sizes, probability, num_partitions)
sr_save_results_HEM = sr.save_results_to_csv(experiment_results_HEM, "GPP_with_HEM")


# Run the experiment for different partitions
size = 300
probability = 0.3
partitions_list = [2, 3, 4, 5]

experiment_results_HEM_diff_par = run_experiment_for_different_partitions(size, probability, partitions_list)

# Save results to CSV for the HEM algorithm
sr.save_results_to_csv(experiment_results_HEM_diff_par, "GPP_with_HEM_different_partition")
