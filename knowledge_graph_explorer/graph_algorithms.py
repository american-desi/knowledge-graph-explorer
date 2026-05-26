"""
Graph algorithms: PageRank, community detection (Louvain-like),
shortest path, centrality measures, connected components, and graph statistics.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from .knowledge_graph import KnowledgeGraph, Node, Edge


def pagerank(graph: KnowledgeGraph, damping: float = 0.85,
             max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, float]:
    """
    Compute PageRank scores for all nodes in the graph.

    Uses the power iteration method with damping factor.
    Treats the graph as undirected for PageRank (considers both directions).
    """
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    if n == 0:
        return {}

    # Initialize scores uniformly
    scores = {nid: 1.0 / n for nid in nodes}

    # Build undirected adjacency for PageRank
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    for _ in range(max_iterations):
        new_scores = {}
        for nid in nodes:
            rank_sum = 0.0
            for neighbor in neighbors[nid]:
                degree = len(neighbors[neighbor])
                if degree > 0:
                    rank_sum += scores[neighbor] / degree
            new_scores[nid] = (1 - damping) / n + damping * rank_sum

        # Check convergence
        diff = sum(abs(new_scores[nid] - scores[nid]) for nid in nodes)
        scores = new_scores
        if diff < tolerance:
            break

    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {nid: s / total for nid, s in scores.items()}

    return scores


def betweenness_centrality(graph: KnowledgeGraph) -> Dict[str, float]:
    """
    Compute betweenness centrality for all nodes.
    Uses Brandes' algorithm on the undirected version of the graph.
    """
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    centrality = {nid: 0.0 for nid in nodes}

    if n < 3:
        return centrality

    # Build undirected adjacency
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    for source in nodes:
        # BFS from source
        stack = []
        predecessors = {nid: [] for nid in nodes}
        sigma = {nid: 0.0 for nid in nodes}
        sigma[source] = 1.0
        dist = {nid: -1 for nid in nodes}
        dist[source] = 0
        queue = [source]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in neighbors[v]:
                # Path discovery
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                # Path counting
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        # Accumulation
        delta = {nid: 0.0 for nid in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != source:
                centrality[w] += delta[w]

    # Normalize
    norm = (n - 1) * (n - 2)
    if norm > 0:
        centrality = {nid: c / norm for nid, c in centrality.items()}

    return centrality


def degree_centrality(graph: KnowledgeGraph) -> Dict[str, float]:
    """Compute degree centrality (normalized degree) for all nodes."""
    n = len(graph.nodes)
    if n <= 1:
        return {nid: 0.0 for nid in graph.nodes}

    centrality = {}
    for nid in graph.nodes:
        degree = graph.get_degree(nid)
        centrality[nid] = degree / (n - 1)

    return centrality


def closeness_centrality(graph: KnowledgeGraph) -> Dict[str, float]:
    """
    Compute closeness centrality for all nodes.
    Uses BFS for shortest paths in unweighted graph.
    """
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    centrality = {nid: 0.0 for nid in nodes}

    if n <= 1:
        return centrality

    # Build undirected adjacency
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    for source in nodes:
        distances = bfs_distances(source, neighbors)
        reachable = {k: v for k, v in distances.items() if v > 0}
        if reachable:
            avg_dist = sum(reachable.values()) / len(reachable)
            centrality[source] = len(reachable) / ((n - 1) * avg_dist) if avg_dist > 0 else 0

    return centrality


def bfs_distances(source: str, neighbors: Dict[str, Set[str]]) -> Dict[str, int]:
    """BFS to compute distances from source to all reachable nodes."""
    distances = {source: 0}
    queue = [source]
    while queue:
        current = queue.pop(0)
        for neighbor in neighbors[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def shortest_path(graph: KnowledgeGraph, source_label: str,
                  target_label: str) -> Optional[List[str]]:
    """
    Find the shortest path between two nodes using BFS.
    Returns list of node IDs or None if no path exists.
    """
    source_id = graph._generate_id(source_label)
    target_id = graph._generate_id(target_label)

    if source_id not in graph.nodes or target_id not in graph.nodes:
        return None

    if source_id == target_id:
        return [source_id]

    # Build undirected adjacency
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    # BFS
    visited = {source_id}
    queue = [(source_id, [source_id])]

    while queue:
        current, path = queue.pop(0)
        for neighbor in neighbors[current]:
            if neighbor == target_id:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path found


def all_shortest_paths(graph: KnowledgeGraph, source_label: str,
                       target_label: str) -> List[List[str]]:
    """Find all shortest paths between two nodes."""
    source_id = graph._generate_id(source_label)
    target_id = graph._generate_id(target_label)

    if source_id not in graph.nodes or target_id not in graph.nodes:
        return []

    if source_id == target_id:
        return [[source_id]]

    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    # BFS to find all shortest paths
    paths = []
    dist = {source_id: 0}
    queue = [(source_id, [source_id])]
    target_dist = None

    while queue:
        current, path = queue.pop(0)

        if target_dist is not None and len(path) > target_dist:
            break

        if current == target_id:
            paths.append(path)
            target_dist = len(path)
            continue

        for neighbor in neighbors[current]:
            new_dist = len(path)
            if neighbor not in dist or dist[neighbor] >= new_dist:
                dist[neighbor] = new_dist
                queue.append((neighbor, path + [neighbor]))

    return paths


def connected_components(graph: KnowledgeGraph) -> List[Set[str]]:
    """Find all connected components in the graph (treating as undirected)."""
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    visited = set()
    components = []

    for node_id in graph.nodes:
        if node_id in visited:
            continue

        # BFS to find component
        component = set()
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    # Sort by size (largest first)
    components.sort(key=len, reverse=True)
    return components


def community_detection_louvain(graph: KnowledgeGraph) -> Dict[str, int]:
    """
    Louvain-like community detection using modularity optimization.
    Returns a mapping of node_id -> community_id.
    """
    nodes = list(graph.nodes.keys())
    if not nodes:
        return {}

    # Build undirected weighted adjacency
    adj = defaultdict(lambda: defaultdict(float))
    for edge in graph.edges:
        adj[edge.source_id][edge.target_id] += edge.weight
        adj[edge.target_id][edge.source_id] += edge.weight

    # Total weight of all edges
    m = sum(edge.weight for edge in graph.edges)
    if m == 0:
        # No edges: each node is its own community
        return {nid: i for i, nid in enumerate(nodes)}

    # Degree of each node (sum of edge weights)
    degree = defaultdict(float)
    for edge in graph.edges:
        degree[edge.source_id] += edge.weight
        degree[edge.target_id] += edge.weight

    # Initialize: each node in its own community
    community = {nid: i for i, nid in enumerate(nodes)}
    num_communities = len(nodes)

    # Phase 1: Local optimization
    improved = True
    max_passes = 20
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        for node in nodes:
            current_comm = community[node]

            # Find neighboring communities and compute modularity gain
            neighbor_comms = defaultdict(float)
            for neighbor, weight in adj[node].items():
                neighbor_comms[community[neighbor]] += weight

            # Remove node from its community
            best_comm = current_comm
            best_gain = 0.0

            ki = degree[node]

            for comm, ki_in in neighbor_comms.items():
                if comm == current_comm:
                    continue

                # Compute modularity gain of moving to this community
                sum_tot = sum(degree[n] for n in nodes if community[n] == comm)
                gain = (ki_in / m) - (sum_tot * ki) / (2 * m * m)

                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm

            if best_comm != current_comm:
                community[node] = best_comm
                improved = True

    # Renumber communities to be contiguous
    unique_comms = sorted(set(community.values()))
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    community = {nid: comm_map[c] for nid, c in community.items()}

    return community


def modularity(graph: KnowledgeGraph, communities: Dict[str, int]) -> float:
    """Compute the modularity of a community partition."""
    m = sum(edge.weight for edge in graph.edges)
    if m == 0:
        return 0.0

    degree = defaultdict(float)
    for edge in graph.edges:
        degree[edge.source_id] += edge.weight
        degree[edge.target_id] += edge.weight

    q = 0.0
    for edge in graph.edges:
        if communities.get(edge.source_id) == communities.get(edge.target_id):
            ki = degree[edge.source_id]
            kj = degree[edge.target_id]
            q += edge.weight - (ki * kj) / (2 * m)

    return q / (2 * m) if m > 0 else 0.0


def graph_diameter(graph: KnowledgeGraph) -> int:
    """Compute the diameter of the largest connected component."""
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    components = connected_components(graph)
    if not components:
        return 0

    largest = components[0]
    max_dist = 0

    for source in largest:
        distances = bfs_distances(source, neighbors)
        if distances:
            max_dist = max(max_dist, max(distances.values()))

    return max_dist


def clustering_coefficient(graph: KnowledgeGraph) -> Dict[str, float]:
    """Compute local clustering coefficient for each node."""
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)

    coefficients = {}
    for nid in graph.nodes:
        nbrs = neighbors[nid]
        k = len(nbrs)
        if k < 2:
            coefficients[nid] = 0.0
            continue

        # Count edges between neighbors
        triangles = 0
        nbr_list = list(nbrs)
        for i, n1 in enumerate(nbr_list):
            for n2 in nbr_list[i + 1:]:
                if n2 in neighbors[n1]:
                    triangles += 1

        possible = k * (k - 1) / 2
        coefficients[nid] = triangles / possible if possible > 0 else 0.0

    return coefficients


def get_comprehensive_statistics(graph: KnowledgeGraph) -> Dict[str, Any]:
    """Compute comprehensive graph statistics and metrics."""
    stats = graph.get_statistics()

    if len(graph.nodes) == 0:
        return stats

    # Centrality measures
    pr = pagerank(graph)
    dc = degree_centrality(graph)

    # Top entities by PageRank
    top_pagerank = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_pagerank"] = [
        {"node": graph.nodes[nid].label, "type": graph.nodes[nid].node_type, "score": score}
        for nid, score in top_pagerank if nid in graph.nodes
    ]

    # Top entities by degree
    top_degree = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_degree_centrality"] = [
        {"node": graph.nodes[nid].label, "type": graph.nodes[nid].node_type, "score": score}
        for nid, score in top_degree if nid in graph.nodes
    ]

    # Connected components
    components = connected_components(graph)
    stats["num_components"] = len(components)
    stats["largest_component_size"] = len(components[0]) if components else 0
    stats["component_sizes"] = [len(c) for c in components[:10]]

    # Communities
    communities = community_detection_louvain(graph)
    num_communities = len(set(communities.values()))
    stats["num_communities"] = num_communities
    stats["modularity"] = modularity(graph, communities)

    # Community sizes
    comm_sizes = defaultdict(int)
    for _, comm in communities.items():
        comm_sizes[comm] += 1
    stats["community_sizes"] = dict(sorted(comm_sizes.items()))

    # Average clustering coefficient
    cc = clustering_coefficient(graph)
    if cc:
        stats["avg_clustering_coefficient"] = sum(cc.values()) / len(cc)

    return stats
