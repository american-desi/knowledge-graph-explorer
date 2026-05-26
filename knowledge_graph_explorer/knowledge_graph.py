"""
Graph data structure with typed nodes and edges, CRUD operations,
graph serialization (JSON export/import), and merge/dedup entities.
"""

import json
from typing import List, Dict, Set, Tuple, Optional, Any
from .entity_extractor import Entity
from .relation_extractor import Relation


class Node:
    """A node in the knowledge graph representing an entity."""

    def __init__(self, node_id: str, label: str, node_type: str,
                 properties: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.label = label
        self.node_type = node_type
        self.properties = properties or {}

    def to_dict(self) -> dict:
        return {
            "id": self.node_id,
            "label": self.label,
            "type": self.node_type,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls(
            node_id=data["id"],
            label=data["label"],
            node_type=data["type"],
            properties=data.get("properties", {}),
        )

    def __repr__(self):
        return f"Node({self.label!r}, type={self.node_type})"

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False

    def __hash__(self):
        return hash(self.node_id)


class Edge:
    """An edge in the knowledge graph representing a relationship."""

    def __init__(self, source_id: str, target_id: str, edge_type: str,
                 weight: float = 1.0, properties: Optional[Dict[str, Any]] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
        self.properties = properties or {}

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type,
            "weight": self.weight,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        return cls(
            source_id=data["source"],
            target_id=data["target"],
            edge_type=data["type"],
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
        )

    def __repr__(self):
        return f"Edge({self.source_id!r} --[{self.edge_type}]--> {self.target_id!r})"

    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.source_id == other.source_id and
                    self.target_id == other.target_id and
                    self.edge_type == other.edge_type)
        return False

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))


class KnowledgeGraph:
    """
    A directed knowledge graph with typed nodes and edges.
    Supports CRUD operations, serialization, and entity deduplication.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._adjacency: Dict[str, List[Edge]] = {}  # outgoing edges
        self._reverse_adjacency: Dict[str, List[Edge]] = {}  # incoming edges
        self._id_counter = 0

    def _generate_id(self, label: str) -> str:
        """Generate a unique node ID from label."""
        # Normalize: lowercase, replace spaces with underscores
        base = label.lower().replace(" ", "_").replace(".", "")
        base = "".join(c for c in base if c.isalnum() or c == "_")
        return base

    # ── Node CRUD ──────────────────────────────────────────────────────────

    def add_node(self, label: str, node_type: str,
                 properties: Optional[Dict[str, Any]] = None) -> Node:
        """Add a node to the graph. Returns existing node if already present."""
        node_id = self._generate_id(label)

        if node_id in self.nodes:
            # Update properties if provided
            if properties:
                self.nodes[node_id].properties.update(properties)
            return self.nodes[node_id]

        node = Node(node_id, label, node_type, properties)
        self.nodes[node_id] = node
        self._adjacency[node_id] = []
        self._reverse_adjacency[node_id] = []
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_node_by_label(self, label: str) -> Optional[Node]:
        """Get a node by its label."""
        node_id = self._generate_id(label)
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self.nodes:
            return False

        # Remove all edges involving this node
        self.edges = [e for e in self.edges
                      if e.source_id != node_id and e.target_id != node_id]

        # Update adjacency lists
        if node_id in self._adjacency:
            del self._adjacency[node_id]
        if node_id in self._reverse_adjacency:
            del self._reverse_adjacency[node_id]

        # Remove from other nodes' adjacency lists
        for nid in self._adjacency:
            self._adjacency[nid] = [e for e in self._adjacency[nid]
                                     if e.target_id != node_id]
        for nid in self._reverse_adjacency:
            self._reverse_adjacency[nid] = [e for e in self._reverse_adjacency[nid]
                                             if e.source_id != node_id]

        del self.nodes[node_id]
        return True

    def update_node(self, node_id: str, **kwargs) -> bool:
        """Update node properties."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        if "label" in kwargs:
            node.label = kwargs["label"]
        if "node_type" in kwargs:
            node.node_type = kwargs["node_type"]
        if "properties" in kwargs:
            node.properties.update(kwargs["properties"])
        return True

    # ── Edge CRUD ──────────────────────────────────────────────────────────

    def add_edge(self, source_label: str, target_label: str, edge_type: str,
                 weight: float = 1.0,
                 properties: Optional[Dict[str, Any]] = None) -> Optional[Edge]:
        """Add an edge between two nodes (by label). Nodes must exist."""
        source_id = self._generate_id(source_label)
        target_id = self._generate_id(target_label)

        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        # Check for duplicate edge
        for edge in self.edges:
            if (edge.source_id == source_id and edge.target_id == target_id
                    and edge.edge_type == edge_type):
                # Update weight if higher
                if weight > edge.weight:
                    edge.weight = weight
                return edge

        edge = Edge(source_id, target_id, edge_type, weight, properties)
        self.edges.append(edge)
        self._adjacency.setdefault(source_id, []).append(edge)
        self._reverse_adjacency.setdefault(target_id, []).append(edge)
        return edge

    def get_edges(self, source_id: Optional[str] = None,
                  target_id: Optional[str] = None,
                  edge_type: Optional[str] = None) -> List[Edge]:
        """Get edges matching the given criteria."""
        results = self.edges
        if source_id:
            results = [e for e in results if e.source_id == source_id]
        if target_id:
            results = [e for e in results if e.target_id == target_id]
        if edge_type:
            results = [e for e in results if e.edge_type == edge_type]
        return results

    def remove_edge(self, source_id: str, target_id: str,
                    edge_type: str) -> bool:
        """Remove a specific edge."""
        for i, edge in enumerate(self.edges):
            if (edge.source_id == source_id and edge.target_id == target_id
                    and edge.edge_type == edge_type):
                self.edges.pop(i)
                # Update adjacency
                self._adjacency[source_id] = [
                    e for e in self._adjacency.get(source_id, [])
                    if not (e.target_id == target_id and e.edge_type == edge_type)
                ]
                self._reverse_adjacency[target_id] = [
                    e for e in self._reverse_adjacency.get(target_id, [])
                    if not (e.source_id == source_id and e.edge_type == edge_type)
                ]
                return True
        return False

    # ── Neighbors ──────────────────────────────────────────────────────────

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Get neighbor node IDs. Direction: 'out', 'in', or 'both'."""
        neighbors = set()
        if direction in ("out", "both"):
            for edge in self._adjacency.get(node_id, []):
                neighbors.add(edge.target_id)
        if direction in ("in", "both"):
            for edge in self._reverse_adjacency.get(node_id, []):
                neighbors.add(edge.source_id)
        return list(neighbors)

    def get_degree(self, node_id: str, direction: str = "both") -> int:
        """Get the degree of a node."""
        return len(self.get_neighbors(node_id, direction))

    # ── Build from extracted data ──────────────────────────────────────────

    def build_from_extractions(self, entities: List[Entity],
                               relations: List[Relation]):
        """Build the graph from extracted entities and relations."""
        # Add entity nodes
        for ent in entities:
            props = {
                "confidence": ent.confidence,
                "mentions": ent.mentions,
            }
            self.add_node(ent.text, ent.entity_type, props)

        # Add relation edges
        for rel in relations:
            # Ensure both nodes exist
            self.add_node(rel.source.text, rel.source.entity_type)
            self.add_node(rel.target.text, rel.target.entity_type)

            props = {
                "confidence": rel.confidence,
                "evidence": rel.evidence[:200] if rel.evidence else "",
            }
            self.add_edge(rel.source.text, rel.target.text,
                          rel.relation_type, weight=rel.confidence, properties=props)

    # ── Merge / Dedup ──────────────────────────────────────────────────────

    def merge_entities(self, keep_id: str, remove_id: str) -> bool:
        """Merge two entity nodes, keeping one and redirecting all edges."""
        if keep_id not in self.nodes or remove_id not in self.nodes:
            return False

        # Redirect edges from remove_id to keep_id
        for edge in self.edges:
            if edge.source_id == remove_id:
                edge.source_id = keep_id
            if edge.target_id == remove_id:
                edge.target_id = keep_id

        # Rebuild adjacency
        self._rebuild_adjacency()

        # Remove the merged node
        del self.nodes[remove_id]
        return True

    def deduplicate_entities(self):
        """Find and merge duplicate/similar entities."""
        # Simple dedup: merge nodes with same normalized ID
        # (This is handled by add_node, but we also check for
        #  substring matches)
        node_ids = list(self.nodes.keys())
        merged = set()

        for i, id1 in enumerate(node_ids):
            if id1 in merged:
                continue
            for id2 in node_ids[i + 1:]:
                if id2 in merged:
                    continue
                # Check if one is a substring of the other
                if id1 in id2 or id2 in id1:
                    # Keep the longer one (more specific)
                    keep = id1 if len(id1) >= len(id2) else id2
                    remove = id2 if keep == id1 else id1
                    self.merge_entities(keep, remove)
                    merged.add(remove)

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize graph to a dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": list(set(n.node_type for n in self.nodes.values())),
                "edge_types": list(set(e.edge_type for e in self.edges)),
            },
        }

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Export graph to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Deserialize graph from a dictionary."""
        graph = cls()
        for node_data in data.get("nodes", []):
            node = Node.from_dict(node_data)
            graph.nodes[node.node_id] = node
            graph._adjacency[node.node_id] = []
            graph._reverse_adjacency[node.node_id] = []

        for edge_data in data.get("edges", []):
            edge = Edge.from_dict(edge_data)
            if edge.source_id in graph.nodes and edge.target_id in graph.nodes:
                graph.edges.append(edge)
                graph._adjacency[edge.source_id].append(edge)
                graph._reverse_adjacency[edge.target_id].append(edge)

        return graph

    @classmethod
    def from_json(cls, filepath: str) -> "KnowledgeGraph":
        """Import graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ── Statistics ─────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if not self.nodes:
            return {"node_count": 0, "edge_count": 0}

        degrees = [self.get_degree(nid) for nid in self.nodes]
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        edge_types = {}
        for edge in self.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "density": (2 * len(self.edges)) / (len(self.nodes) * (len(self.nodes) - 1))
            if len(self.nodes) > 1 else 0,
            "isolated_nodes": sum(1 for d in degrees if d == 0),
        }

    def _rebuild_adjacency(self):
        """Rebuild adjacency lists from edges."""
        self._adjacency = {nid: [] for nid in self.nodes}
        self._reverse_adjacency = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            if edge.source_id in self.nodes and edge.target_id in self.nodes:
                self._adjacency[edge.source_id].append(edge)
                self._reverse_adjacency[edge.target_id].append(edge)

    def __repr__(self):
        return f"KnowledgeGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
