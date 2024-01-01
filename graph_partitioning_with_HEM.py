import random
import networkx as nx
import save_results as sr
import pandas as pd


def assign_default_weights(G):
    for u, v in G.edges():
        G[u][v]['weight'] = 1


def heavy_edge_matching(g):
    matched = set()
    matchings = []

    for u, v, data in sorted(g.edges(data=True), key=lambda x: x[2]['weight'], reverse=True):
        if u not in matched and v not in matched:
            matchings.append((u, v))
            matched.update([u, v])

    return matchings


def coarsen_graph(G, matchings):
    coarse_G = nx.Graph()
    mapping = {}

    for u, v in matchings:
        coarse_node = len(coarse_G.nodes())
        coarse_G.add_node(coarse_node)
        mapping[u] = coarse_node
        mapping[v] = coarse_node

    unmatched_nodes = set(G.nodes()) - set(mapping.keys())
    for node in unmatched_nodes:
        coarse_node = len(coarse_G.nodes())
        coarse_G.add_node(coarse_node)
        mapping[node] = coarse_node

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

    for i, node in enumerate(nodes):
        partitions[i % num_partitions].add(node)
    return partitions

def calculate_initial_edge_cuts(G):
    edge_cuts = {}
    for u, v in G.edges():
        if G.nodes[u]['partition'] != G.nodes[v]['partition']:
            edge_cuts[(u, v)] = edge_cuts.get((u, v), 0) + 1
    return edge_cuts


def update_edge_cuts(G, edge_cuts, node):
    for neighbor in G.neighbors(node):
        if G.nodes[node]['partition'] != G.nodes[neighbor]['partition']:
            edge_cuts[(node, neighbor)] = edge_cuts.get((node, neighbor), 0) + 1
        else:
            edge_cuts.pop((node, neighbor), None)


def get_node_edge_cut_contribution(G, node):
    contribution = 0
    for neighbor in G.neighbors(node):
        if G.nodes[node]['partition'] != G.nodes[neighbor]['partition']:
            contribution += 1
    return contribution


def refine_partitions(G, partitions, num_iterations=10):
    edge_cuts = calculate_initial_edge_cuts(G)

    for _ in range(num_iterations):
        for part_id, nodes in list(partitions.items()):
            for node in list(nodes):
                if len(nodes) > 1:  # Ensure not emptying a partition
                    best_move = (None, len(edge_cuts))
                    current_contribution = get_node_edge_cut_contribution(G, node)

                    for other_part_id in partitions:
                        if other_part_id != part_id:
                            partitions[part_id].remove(node)
                            partitions[other_part_id].add(node)
                            G.nodes[node]['partition'] = other_part_id
                            update_edge_cuts(G, edge_cuts, node)

                            new_contribution = get_node_edge_cut_contribution(G, node)
                            if new_contribution < current_contribution:
                                best_move = (other_part_id, len(edge_cuts))

                            partitions[other_part_id].remove(node)
                            partitions[part_id].add(node)
                            G.nodes[node]['partition'] = part_id
                            update_edge_cuts(G, edge_cuts, node)

                    if best_move[0] is not None:
                        partitions[part_id].remove(node)
                        partitions[best_move[0]].add(node)
                        G.nodes[node]['partition'] = best_move[0]
                        update_edge_cuts(G, edge_cuts, node)

    return partitions


def uncoarsen_and_refine(partitions, mapping):
    inverse_mapping = {}
    for orig_node, coarse_node in mapping.items():
        inverse_mapping.setdefault(coarse_node, []).append(orig_node)

    original_partitions = {i: set() for i in partitions}

    for part_id, coarse_nodes in partitions.items():
        for coarse_node in coarse_nodes:
            original_nodes = inverse_mapping[coarse_node]
            original_partitions[part_id].update(original_nodes)

    return original_partitions


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


def multiway_partition(G, num_partitions):
    assign_default_weights(G)
    matchings = heavy_edge_matching(G)
    coarse_G, mapping = coarsen_graph(G, matchings)
    partitions = partition_graph_balanced(coarse_G, num_partitions)

    for part_id, nodes in partitions.items():
        for node in nodes:
            coarse_G.nodes[node]['partition'] = part_id

    original_partitions = uncoarsen_and_refine(partitions, mapping)
    for part_id, nodes in original_partitions.items():
        for node in nodes:
            G.nodes[node]['partition'] = part_id

    refined_partitions = refine_partitions(G, original_partitions)
    edge_cuts = calculate_edge_cuts(G, refined_partitions)
    return refined_partitions, edge_cuts


# Test the algorithm
# G = nx.erdos_renyi_graph(500, 0.3)
# num_partitions = 5
# partitions, edge_cuts = multiway_partition(G, num_partitions)
# print(f"Partitions: {partitions}")
# print(f"Edge cuts: {edge_cuts}")


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
sizes = [100, 200, 300, 400, 500]
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

