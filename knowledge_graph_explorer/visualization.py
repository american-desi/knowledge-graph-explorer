"""
Force-directed graph layout from scratch with matplotlib-based rendering.
Colored nodes by type, labeled edges, adjustable layout parameters.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .knowledge_graph import KnowledgeGraph, Node, Edge


# Color scheme for entity types
TYPE_COLORS = {
    "PERSON": "#4CAF50",        # Green
    "ORGANIZATION": "#2196F3",  # Blue
    "LOCATION": "#FF9800",      # Orange
    "DATE": "#9C27B0",          # Purple
    "TECHNOLOGY": "#F44336",    # Red
}

DEFAULT_COLOR = "#607D8B"  # Gray

# Edge colors by relation type
EDGE_COLORS = {
    "founded_by": "#2E7D32",
    "located_in": "#E65100",
    "works_for": "#1565C0",
    "acquired": "#B71C1C",
    "developed": "#4A148C",
    "born_in": "#FF6F00",
    "partner_of": "#00695C",
    "competes_with": "#880E4F",
}

DEFAULT_EDGE_COLOR = "#9E9E9E"


class ForceDirectedLayout:
    """
    Force-directed graph layout using Fruchterman-Reingold algorithm.
    Implemented from scratch without networkx layout functions.
    """

    def __init__(self, graph: KnowledgeGraph, width: float = 10.0,
                 height: float = 10.0, seed: int = 42):
        self.graph = graph
        self.width = width
        self.height = height
        self.seed = seed
        self.positions: Dict[str, Tuple[float, float]] = {}

    def compute_layout(self, iterations: int = 150,
                       repulsion: float = 1.0,
                       attraction: float = 0.01,
                       gravity: float = 0.05,
                       cooling: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Compute force-directed layout positions for all nodes.

        Parameters:
            iterations: Number of simulation steps
            repulsion: Strength of repulsive force between nodes
            attraction: Strength of attractive force along edges
            gravity: Pull toward center
            cooling: Temperature reduction factor per iteration
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        nodes = list(self.graph.nodes.keys())
        n = len(nodes)

        if n == 0:
            return {}

        if n == 1:
            self.positions = {nodes[0]: (self.width / 2, self.height / 2)}
            return self.positions

        # Build adjacency for fast lookup
        neighbors = defaultdict(set)
        for edge in self.graph.edges:
            neighbors[edge.source_id].add(edge.target_id)
            neighbors[edge.target_id].add(edge.source_id)

        # Initialize positions randomly
        pos = np.random.rand(n, 2) * [self.width, self.height]
        node_idx = {nid: i for i, nid in enumerate(nodes)}

        # Optimal distance
        area = self.width * self.height
        k = math.sqrt(area / n) * 0.8

        # Temperature (limits movement per iteration)
        temp = self.width / 5

        for iteration in range(iterations):
            forces = np.zeros((n, 2))

            # Repulsive forces between all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    delta = pos[i] - pos[j]
                    dist = np.linalg.norm(delta)
                    if dist < 0.01:
                        dist = 0.01
                        delta = np.random.rand(2) * 0.1

                    # Coulomb-like repulsion
                    force_mag = repulsion * (k * k) / dist
                    force = (delta / dist) * force_mag

                    forces[i] += force
                    forces[j] -= force

            # Attractive forces along edges
            for edge in self.graph.edges:
                if edge.source_id in node_idx and edge.target_id in node_idx:
                    i = node_idx[edge.source_id]
                    j = node_idx[edge.target_id]
                    delta = pos[j] - pos[i]
                    dist = np.linalg.norm(delta)
                    if dist < 0.01:
                        continue

                    # Hooke's law attraction
                    force_mag = attraction * (dist * dist) / k
                    force = (delta / dist) * force_mag

                    forces[i] += force
                    forces[j] -= force

            # Gravity toward center
            center = np.array([self.width / 2, self.height / 2])
            for i in range(n):
                delta = center - pos[i]
                dist = np.linalg.norm(delta)
                if dist > 0.01:
                    forces[i] += gravity * delta

            # Apply forces with temperature limit
            for i in range(n):
                force_mag = np.linalg.norm(forces[i])
                if force_mag > 0:
                    # Limit displacement by temperature
                    displacement = forces[i] / force_mag * min(force_mag, temp)
                    pos[i] += displacement

                # Keep within bounds (with padding)
                pad = 0.5
                pos[i][0] = max(pad, min(self.width - pad, pos[i][0]))
                pos[i][1] = max(pad, min(self.height - pad, pos[i][1]))

            # Cool down
            temp *= cooling

        # Store results
        self.positions = {nodes[i]: (float(pos[i][0]), float(pos[i][1]))
                          for i in range(n)}
        return self.positions


def visualize_graph(graph: KnowledgeGraph,
                    output_path: str = "knowledge_graph.png",
                    figsize: Tuple[int, int] = (20, 16),
                    title: str = "Knowledge Graph",
                    show_edge_labels: bool = True,
                    node_size_factor: float = 1.0,
                    font_size: int = 8,
                    layout_iterations: int = 200,
                    dpi: int = 150) -> Optional[str]:
    """
    Visualize the knowledge graph with force-directed layout.

    Returns the output file path, or None if matplotlib is not available.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None

    if len(graph.nodes) == 0:
        print("Empty graph. Nothing to visualize.")
        return None

    print(f"  Computing force-directed layout ({layout_iterations} iterations)...")
    layout = ForceDirectedLayout(graph, width=10.0, height=10.0)
    positions = layout.compute_layout(iterations=layout_iterations)

    if not positions:
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.5, 11.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw edges first (behind nodes)
    print("  Drawing edges...")
    edge_labels_drawn = set()
    for edge in graph.edges:
        if edge.source_id not in positions or edge.target_id not in positions:
            continue

        x1, y1 = positions[edge.source_id]
        x2, y2 = positions[edge.target_id]

        color = EDGE_COLORS.get(edge.edge_type, DEFAULT_EDGE_COLOR)
        alpha = min(0.8, 0.3 + edge.weight * 0.5)

        # Draw edge as line with arrow
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.01:
            continue

        # Shorten line to not overlap with nodes
        shrink = 0.25
        sx1 = x1 + (dx / dist) * shrink
        sy1 = y1 + (dy / dist) * shrink
        sx2 = x2 - (dx / dist) * shrink
        sy2 = y2 - (dy / dist) * shrink

        ax.annotate(
            "", xy=(sx2, sy2), xytext=(sx1, sy1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=alpha,
                lw=1.2,
                connectionstyle="arc3,rad=0.1",
            ),
        )

        # Draw edge label
        if show_edge_labels:
            label_key = (edge.source_id, edge.target_id, edge.edge_type)
            if label_key not in edge_labels_drawn:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                # Offset label slightly
                offset = 0.15
                mid_x += offset * (-dy / dist) if dist > 0 else 0
                mid_y += offset * (dx / dist) if dist > 0 else 0

                ax.text(
                    mid_x, mid_y,
                    edge.edge_type.replace("_", " "),
                    fontsize=max(5, font_size - 2),
                    color=color,
                    alpha=0.7,
                    ha='center', va='center',
                    fontweight='light',
                    style='italic',
                    bbox=dict(boxstyle='round,pad=0.1',
                              facecolor='white',
                              edgecolor='none',
                              alpha=0.6),
                )
                edge_labels_drawn.add(label_key)

    # Draw nodes
    print("  Drawing nodes...")
    # Compute node sizes based on degree
    degrees = {nid: graph.get_degree(nid) for nid in graph.nodes}
    max_degree = max(degrees.values()) if degrees else 1

    for node_id, (x, y) in positions.items():
        node = graph.nodes.get(node_id)
        if not node:
            continue

        color = TYPE_COLORS.get(node.node_type, DEFAULT_COLOR)
        degree = degrees.get(node_id, 0)

        # Size proportional to degree
        base_size = 300
        size = base_size + (degree / max(max_degree, 1)) * 800
        size *= node_size_factor

        # Draw node circle
        ax.scatter(
            x, y,
            s=size,
            c=color,
            alpha=0.85,
            edgecolors='white',
            linewidths=1.5,
            zorder=3,
        )

        # Draw node label
        ax.text(
            x, y - 0.22,
            node.label,
            fontsize=font_size,
            ha='center', va='top',
            fontweight='bold',
            color='#333333',
            zorder=4,
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor='white',
                      edgecolor='none',
                      alpha=0.7),
        )

    # Legend for node types
    legend_patches = []
    present_types = set(n.node_type for n in graph.nodes.values())
    for node_type in sorted(present_types):
        color = TYPE_COLORS.get(node_type, DEFAULT_COLOR)
        count = sum(1 for n in graph.nodes.values() if n.node_type == node_type)
        patch = mpatches.Patch(color=color, label=f"{node_type} ({count})")
        legend_patches.append(patch)

    ax.legend(
        handles=legend_patches,
        loc='upper left',
        fontsize=10,
        framealpha=0.9,
        edgecolor='#cccccc',
        title="Entity Types",
        title_fontsize=11,
    )

    # Title and stats
    stats_text = (
        f"Nodes: {len(graph.nodes)} | "
        f"Edges: {len(graph.edges)} | "
        f"Types: {len(present_types)}"
    )
    ax.set_title(
        f"{title}\n{stats_text}",
        fontsize=16,
        fontweight='bold',
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  Graph visualization saved to: {output_path}")
    return output_path


def visualize_communities(graph: KnowledgeGraph,
                          communities: Dict[str, int],
                          output_path: str = "communities.png",
                          figsize: Tuple[int, int] = (20, 16),
                          title: str = "Community Structure",
                          dpi: int = 150) -> Optional[str]:
    """Visualize the graph colored by community membership."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None

    if len(graph.nodes) == 0:
        return None

    print(f"  Computing layout for community visualization...")
    layout = ForceDirectedLayout(graph, width=10.0, height=10.0, seed=42)
    positions = layout.compute_layout(iterations=200)

    if not positions:
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.5, 11.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Generate community colors
    num_communities = max(communities.values()) + 1 if communities else 0
    try:
        cmap = plt.colormaps.get_cmap('tab20').resampled(max(num_communities, 1))
    except AttributeError:
        cmap = plt.cm.get_cmap('tab20', max(num_communities, 1))
    comm_colors = {i: cmap(i) for i in range(num_communities)}

    # Draw edges
    for edge in graph.edges:
        if edge.source_id not in positions or edge.target_id not in positions:
            continue
        x1, y1 = positions[edge.source_id]
        x2, y2 = positions[edge.target_id]

        same_comm = communities.get(edge.source_id) == communities.get(edge.target_id)
        alpha = 0.4 if same_comm else 0.15
        color = '#666666' if same_comm else '#cccccc'

        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, lw=0.8, zorder=1)

    # Draw nodes colored by community
    degrees = {nid: graph.get_degree(nid) for nid in graph.nodes}
    max_degree = max(degrees.values()) if degrees else 1

    for node_id, (x, y) in positions.items():
        node = graph.nodes.get(node_id)
        if not node:
            continue

        comm = communities.get(node_id, 0)
        color = comm_colors.get(comm, '#999999')
        degree = degrees.get(node_id, 0)
        size = 200 + (degree / max(max_degree, 1)) * 600

        ax.scatter(x, y, s=size, c=[color], alpha=0.85,
                   edgecolors='white', linewidths=1.2, zorder=3)

        ax.text(x, y - 0.2, node.label, fontsize=7,
                ha='center', va='top', fontweight='bold',
                color='#333333', zorder=4,
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='white', edgecolor='none', alpha=0.6))

    # Legend
    legend_patches = []
    comm_sizes = defaultdict(int)
    for _, comm in communities.items():
        comm_sizes[comm] += 1

    for comm_id in sorted(comm_sizes.keys()):
        color = comm_colors.get(comm_id, '#999999')
        patch = mpatches.Patch(
            color=color,
            label=f"Community {comm_id} ({comm_sizes[comm_id]} entities)"
        )
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc='upper left', fontsize=9,
              framealpha=0.9, title="Communities", title_fontsize=10)

    ax.set_title(
        f"{title}\n{num_communities} communities detected | "
        f"Nodes: {len(graph.nodes)}",
        fontsize=16, fontweight='bold', pad=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  Community visualization saved to: {output_path}")
    return output_path


def visualize_subgraph(graph: KnowledgeGraph,
                       center_label: str,
                       max_depth: int = 2,
                       output_path: str = "subgraph.png",
                       figsize: Tuple[int, int] = (16, 12),
                       dpi: int = 150) -> Optional[str]:
    """Visualize a subgraph centered on a specific entity."""
    if not HAS_MATPLOTLIB:
        return None

    center_id = graph._generate_id(center_label)
    if center_id not in graph.nodes:
        print(f"Entity '{center_label}' not found in graph.")
        return None

    # BFS to find nodes within max_depth
    visited = {center_id}
    frontier = [center_id]
    for _ in range(max_depth):
        next_frontier = []
        for nid in frontier:
            for neighbor_id in graph.get_neighbors(nid):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
        frontier = next_frontier

    # Build subgraph
    sub = KnowledgeGraph()
    for nid in visited:
        node = graph.nodes[nid]
        sub.add_node(node.label, node.node_type, node.properties)

    for edge in graph.edges:
        if edge.source_id in visited and edge.target_id in visited:
            src = graph.nodes[edge.source_id]
            tgt = graph.nodes[edge.target_id]
            sub.add_edge(src.label, tgt.label, edge.edge_type, edge.weight)

    return visualize_graph(
        sub, output_path=output_path, figsize=figsize,
        title=f"Subgraph around '{center_label}' (depth={max_depth})",
        dpi=dpi
    )
