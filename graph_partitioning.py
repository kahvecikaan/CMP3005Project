class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]
        self.vertex_weight = [1] * vertices  # Initially, each vertex has a weight of 1

    def add_edge(self, u, v, weight=1):
        if v not in [edge[0] for edge in self.graph[u]]:
            self.graph[u].append((v, weight))
        if u not in [edge[0] for edge in self.graph[v]]:
            self.graph[v].append((u, weight))

    def show(self):
        for i in range(self.V):
            print(f"{i} --> {self.graph[i]}")


def calculate_edge_cut(graph, partitions):
    edge_cut = 0
    for i in range(graph.V):
        for (j, _) in graph.graph[i]:
            if partitions[i] != partitions[j]:
                edge_cut += 1
    return edge_cut // 2  # Each edge is counted twice


def merge_vertices(graph, v1, v2):
    for v, w in graph.graph[v2]:
        if v != v1:
            graph.add_edge(v1, v, w)
    graph.vertex_weight[v1] += graph.vertex_weight[v2]
    graph.vertex_weight[v2] = 0
    graph.graph[v2] = []


def heavy_edge_matching(graph):
    matched = [False] * graph.V
    for u in range(graph.V):
        if not matched[u]:
            max_weight = -1
            selected_vertex = -1
            for v, weight in graph.graph[u]:
                if not matched[v] and weight > max_weight:
                    max_weight = weight
                    selected_vertex = v
            if selected_vertex != -1:
                merge_vertices(graph, u, selected_vertex)
                matched[u] = True
                matched[selected_vertex] = True


def partition(graph, k):
    if k == 1:
        return [0] * graph.V
    # bisection only
    # Splitting the vertices based on their weights
    sorted_vertices = sorted(range(graph.V), key=lambda x: graph.vertex_weight[x], reverse=True)
    mid = len(sorted_vertices) // 2
    partitions = [0] * graph.V
    for i in range(mid):
        partitions[sorted_vertices[i]] = 1

    return partitions


def kernighan_lin_refinement(graph, partitions):
    improvement = True
    while improvement:
        improvement = False
        for u in range(graph.V):
            for v in range(graph.V):
                if partitions[u] != partitions[v]:
                    current_cut = calculate_edge_cut(graph, partitions)
                    partitions[u], partitions[v] = partitions[v], partitions[u]  # Swap
                    new_cut = calculate_edge_cut(graph, partitions)
                    if new_cut < current_cut:
                        improvement = True
                    else:
                        partitions[u], partitions[v] = partitions[v], partitions[u]  # Swap back if no improvement
    return partitions


def graph_partitioning(graph, k):
    # Coarsening
    heavy_edge_matching(graph)

    # Partitioning
    partitions = partition(graph, k)

    # Uncoarsening and Refinement
    refined_partitions = kernighan_lin_refinement(graph, partitions)

    return refined_partitions


# Create a simple graph for testing
g = Graph(8)
g.add_edge(0, 7)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)
g.add_edge(4, 5)
g.add_edge(5, 6)
g.add_edge(6, 7)
g.add_edge(7, 2)
g.add_edge(1, 4)
g.add_edge(2, 5)
g.add_edge(3, 6)
g.add_edge(7, 0)

# Show the graph
# g.show()

# Partition the graph into 2 parts
# partition = graph_partitioning(g, 2)
# print(partition)


def k_way_partition(graph, k, start=0, partition_map=None):
    """
    Recursive function to perform k-way partitioning of the graph.
    """
    if partition_map is None:
        partition_map = [0] * graph.V

    if k == 1:
        for i in range(start, graph.V):
            partition_map[i] = start
        return partition_map

    if k == 0:
        return partition_map

    # Split the current segment into two parts
    mid = (start + graph.V) // 2
    for i in range(start, mid):
        partition_map[i] = start

    # Recursive calls for the two halves
    k_way_partition(graph, k - 1, start + 1, partition_map)

    return partition_map


def graph_partitioning_k_way(graph, k):
    """
    Function to partition the graph into k subsets.
    """
    # Coarsening
    heavy_edge_matching(graph)

    # k-way Partitioning
    partitions = k_way_partition(graph, k)

    # Uncoarsening and Refinement
    refined_partitions = kernighan_lin_refinement(graph, partitions)

    return refined_partitions


# Perform graph partitioning into 3 subsets
# partitions_three_subsets = graph_partitioning_k_way(g, 3)
# print(partitions_three_subsets)

g2 = Graph(6)

g2.add_edge(0, 1)
g2.add_edge(0, 3)
g2.add_edge(1, 2)
g2.add_edge(3, 2)
g2.add_edge(3, 4)
g2.add_edge(1, 5)
g2.add_edge(2, 5)
g2.show()

print(partition(g2, 2))

# The following function is not correct. You need to fix it.


def generate_random_graph(n, m):
    """
    Function to generate a random graph with n vertices and m edges.
    """
    import random
    random_graph = Graph(n)
    edges = []
    for _ in range(m):
        u, v = random.randint(0, n - 1), random.randint(0, n - 1)
        if (u, v) not in edges and u != v:
            random_graph.add_edge(u, v)
            edges.append((u, v))
    return random_graph


g3 = generate_random_graph(10, 20)
g3.show()
print(partition(g3, 2))
