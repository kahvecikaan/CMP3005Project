import random
import networkx as nx
import save_results as sr
import pandas as pd


def assign_default_weights(g):
    for u, v in g.edges():
        g[u][v]['weight'] = 1


def heavy_edge_matching(g):
    matched = set()
    matchings = []

    for u, v, data in sorted(g.edges(data=True), key=lambda x: x[2]['weight'], reverse=True):
        if u not in matched and v not in matched:
            matchings.append((u, v))
            matched.update([u, v])

    return matchings


def coarsen_graph(g, matchings):
    coarse_g = nx.Graph()
    mapping = {}

    for u, v in matchings:
        coarse_node = len(coarse_g.nodes())
        coarse_g.add_node(coarse_node)
        mapping[u] = coarse_node
        mapping[v] = coarse_node

    unmatched_nodes = set(g.nodes()) - set(mapping.keys())
    for node in unmatched_nodes:
        coarse_node = len(coarse_g.nodes())
        coarse_g.add_node(coarse_node)
        mapping[node] = coarse_node

    for u, v, data in g.edges(data=True):
        new_u, new_v = mapping[u], mapping[v]
        if coarse_g.has_edge(new_u, new_v):
            coarse_g[new_u][new_v]['weight'] += data['weight']
        else:
            coarse_g.add_edge(new_u, new_v, weight=data['weight'])

    return coarse_g, mapping


def partition_graph_balanced(g, num_partitions):
    partitions = {i: set() for i in range(num_partitions)}
    nodes = list(g.nodes())
    random.shuffle(nodes)

    for i, node in enumerate(nodes):
        partitions[i % num_partitions].add(node)
    return partitions


def calculate_initial_edge_cuts(g):
    edge_cuts = {}
    for u, v in g.edges():
        if g.nodes[u]['partition'] != g.nodes[v]['partition']:
            edge_cuts[(u, v)] = edge_cuts.get((u, v), 0) + 1
    return edge_cuts


def update_edge_cuts(g, edge_cuts, node):
    for neighbor in g.neighbors(node):
        if g.nodes[node]['partition'] != g.nodes[neighbor]['partition']:
            edge_cuts[(node, neighbor)] = edge_cuts.get((node, neighbor), 0) + 1
        else:
            edge_cuts.pop((node, neighbor), None)


def get_node_edge_cut_contribution(g, node):
    contribution = 0
    for neighbor in g.neighbors(node):
        if g.nodes[node]['partition'] != g.nodes[neighbor]['partition']:
            contribution += 1
    return contribution


def refine_partitions(g, partitions, num_iterations=10):
    edge_cuts = calculate_initial_edge_cuts(g)

    for _ in range(num_iterations):
        for part_id, nodes in list(partitions.items()):
            for node in list(nodes):
                if len(nodes) > 1:  # Ensure not emptying a partition
                    best_move = (None, len(edge_cuts))
                    current_contribution = get_node_edge_cut_contribution(g, node)

                    for other_part_id in partitions:
                        if other_part_id != part_id:
                            partitions[part_id].remove(node)
                            partitions[other_part_id].add(node)
                            g.nodes[node]['partition'] = other_part_id
                            update_edge_cuts(g, edge_cuts, node)

                            new_contribution = get_node_edge_cut_contribution(g, node)
                            if new_contribution < current_contribution:
                                best_move = (other_part_id, len(edge_cuts))

                            partitions[other_part_id].remove(node)
                            partitions[part_id].add(node)
                            g.nodes[node]['partition'] = part_id
                            update_edge_cuts(g, edge_cuts, node)

                    if best_move[0] is not None:
                        partitions[part_id].remove(node)
                        partitions[best_move[0]].add(node)
                        g.nodes[node]['partition'] = best_move[0]
                        update_edge_cuts(g, edge_cuts, node)

    return partitions


def uncoarsen_graph(partitions, mapping):
    inverse_mapping = {}
    for orig_node, coarse_node in mapping.items():
        inverse_mapping.setdefault(coarse_node, []).append(orig_node)

    original_partitions = {i: set() for i in partitions}

    for part_id, coarse_nodes in partitions.items():
        for coarse_node in coarse_nodes:
            original_nodes = inverse_mapping[coarse_node]
            original_partitions[part_id].update(original_nodes)

    return original_partitions


def calculate_edge_cuts(g, partitions):
    edge_cuts = 0
    seen_edges = set()

    for u, v in g.edges():
        if (u, v) not in seen_edges and (v, u) not in seen_edges:
            for part in partitions.values():
                if (u in part and v not in part) or (v in part and u not in part):
                    edge_cuts += 1
                    seen_edges.add((u, v))
                    break

    return edge_cuts


def multiway_partition(g, num_partitions):
    assign_default_weights(g)
    matchings = heavy_edge_matching(g)
    coarse_G, mapping = coarsen_graph(g, matchings)
    partitions = partition_graph_balanced(coarse_G, num_partitions)

    for part_id, nodes in partitions.items():
        for node in nodes:
            coarse_G.nodes[node]['partition'] = part_id

    original_partitions = uncoarsen_graph(partitions, mapping)
    for part_id, nodes in original_partitions.items():
        for node in nodes:
            g.nodes[node]['partition'] = part_id

    refined_partitions = refine_partitions(g, original_partitions)
    edge_cuts = calculate_edge_cuts(g, refined_partitions)
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
        g_er = nx.erdos_renyi_graph(n=size, p=p)
        _, edge_cuts = multiway_partition(g_er, k)
        results["Graph Size"].append(size)
        results["Edge Cuts"].append(edge_cuts)
    return pd.DataFrame(results)


def run_experiment_for_different_partitions(size, p, partitions_list):
    results = {"Number of Partitions": [], "Edge Cuts": []}
    g_er = nx.erdos_renyi_graph(n=size, p=p)
    for k in partitions_list:
        _, edge_cuts = multiway_partition(g_er, k)
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

