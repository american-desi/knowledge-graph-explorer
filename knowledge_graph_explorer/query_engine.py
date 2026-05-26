"""
Simple query language for graph traversal.
Supports natural-ish queries like:
  - "find all entities connected to Google"
  - "path between Elon Musk and Tesla"
  - "entities of type PERSON who are connected to ORGANIZATION"
  - "relations of type founded_by"
  - "neighbors of Google"
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from .knowledge_graph import KnowledgeGraph, Node, Edge
from .graph_algorithms import shortest_path


class QueryResult:
    """Represents the result of a query."""

    def __init__(self, query: str, result_type: str, data: Any,
                 description: str = ""):
        self.query = query
        self.result_type = result_type  # "nodes", "edges", "path", "stats"
        self.data = data
        self.description = description

    def __repr__(self):
        return f"QueryResult(type={self.result_type}, items={self._count()})"

    def _count(self) -> int:
        if isinstance(self.data, list):
            return len(self.data)
        if isinstance(self.data, dict):
            return len(self.data)
        return 1

    def to_string(self) -> str:
        """Format the result as a readable string."""
        lines = [f"Query: {self.query}"]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  Result type: {self.result_type}")

        if self.result_type == "nodes":
            lines.append(f"  Found {len(self.data)} node(s):")
            for node in self.data[:20]:
                if isinstance(node, Node):
                    lines.append(f"    - {node.label} [{node.node_type}]")
                elif isinstance(node, dict):
                    lines.append(f"    - {node.get('label', '?')} [{node.get('type', '?')}]")

        elif self.result_type == "edges":
            lines.append(f"  Found {len(self.data)} relation(s):")
            for edge in self.data[:20]:
                if isinstance(edge, Edge):
                    lines.append(f"    - {edge.source_id} --[{edge.edge_type}]--> {edge.target_id}")
                elif isinstance(edge, dict):
                    lines.append(
                        f"    - {edge.get('source', '?')} --[{edge.get('type', '?')}]--> "
                        f"{edge.get('target', '?')}"
                    )

        elif self.result_type == "path":
            if self.data:
                path_str = " -> ".join(self.data)
                lines.append(f"  Path ({len(self.data)} nodes): {path_str}")
            else:
                lines.append("  No path found.")

        elif self.result_type == "stats":
            for key, value in self.data.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)


class QueryEngine:
    """Query engine for the knowledge graph with natural-ish syntax."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def execute(self, query: str) -> QueryResult:
        """Parse and execute a query string."""
        query = query.strip()
        lower = query.lower()

        # Try each query pattern
        if lower.startswith("find all") or lower.startswith("find entities"):
            return self._handle_find(query)

        elif lower.startswith("path between") or lower.startswith("find path"):
            return self._handle_path(query)

        elif lower.startswith("neighbors of") or lower.startswith("connections of"):
            return self._handle_neighbors(query)

        elif lower.startswith("relations of type") or lower.startswith("edges of type"):
            return self._handle_relation_type(query)

        elif lower.startswith("entities of type"):
            return self._handle_entity_type(query)

        elif lower.startswith("info about") or lower.startswith("details of"):
            return self._handle_info(query)

        elif lower.startswith("count"):
            return self._handle_count(query)

        elif lower.startswith("stats") or lower.startswith("statistics"):
            return self._handle_stats(query)

        elif lower.startswith("connected to"):
            return self._handle_connected_to(query)

        else:
            # Try to interpret as a general search
            return self._handle_general_search(query)

    def _handle_find(self, query: str) -> QueryResult:
        """Handle 'find all entities connected to X' style queries."""
        lower = query.lower()

        # "find all entities connected to X"
        match = re.search(r'connected\s+to\s+(.+?)(?:\s+of\s+type\s+(\w+))?$',
                          lower, re.IGNORECASE)
        if match:
            entity_name = match.group(1).strip().strip('"\'')
            filter_type = match.group(2)
            return self._find_connected(entity_name, filter_type)

        # "find all PERSON" or "find entities of type PERSON"
        match = re.search(r'(?:find\s+(?:all\s+)?(?:entities\s+)?(?:of\s+type\s+)?)'
                          r'(\w+)(?:\s+who\s+are\s+connected\s+to\s+(\w+))?',
                          query, re.IGNORECASE)
        if match:
            entity_type = match.group(1).upper()
            connected_type = match.group(2)
            return self._find_by_type(entity_type, connected_type)

        return QueryResult(query, "nodes", [], "Could not parse query.")

    def _handle_path(self, query: str) -> QueryResult:
        """Handle 'path between X and Y' queries."""
        # Extract entity names
        match = re.search(
            r'(?:path\s+between|find\s+path\s+(?:from\s+)?)'
            r'(.+?)\s+(?:and|to)\s+(.+?)$',
            query, re.IGNORECASE
        )
        if not match:
            return QueryResult(query, "path", None, "Could not parse path query.")

        source = match.group(1).strip().strip('"\'')
        target = match.group(2).strip().strip('"\'')

        path = shortest_path(self.graph, source, target)
        if path:
            # Convert IDs to labels
            path_labels = []
            for nid in path:
                node = self.graph.get_node(nid)
                path_labels.append(node.label if node else nid)
            return QueryResult(query, "path", path_labels,
                               f"Shortest path from '{source}' to '{target}'")
        else:
            return QueryResult(query, "path", [],
                               f"No path found between '{source}' and '{target}'")

    def _handle_neighbors(self, query: str) -> QueryResult:
        """Handle 'neighbors of X' queries."""
        match = re.search(r'(?:neighbors|connections)\s+of\s+(.+?)$',
                          query, re.IGNORECASE)
        if not match:
            return QueryResult(query, "nodes", [], "Could not parse neighbor query.")

        entity_name = match.group(1).strip().strip('"\'')
        node_id = self.graph._generate_id(entity_name)

        if node_id not in self.graph.nodes:
            # Try fuzzy match
            node_id = self._fuzzy_find_node(entity_name)

        if not node_id:
            return QueryResult(query, "nodes", [],
                               f"Entity '{entity_name}' not found.")

        neighbor_ids = self.graph.get_neighbors(node_id)
        nodes = [self.graph.nodes[nid] for nid in neighbor_ids
                 if nid in self.graph.nodes]

        return QueryResult(query, "nodes", nodes,
                           f"Neighbors of '{entity_name}'")

    def _handle_relation_type(self, query: str) -> QueryResult:
        """Handle 'relations of type X' queries."""
        match = re.search(r'(?:relations|edges)\s+of\s+type\s+(\w+)',
                          query, re.IGNORECASE)
        if not match:
            return QueryResult(query, "edges", [], "Could not parse relation type query.")

        rel_type = match.group(1).lower()
        edges = self.graph.get_edges(edge_type=rel_type)

        # Enrich with node labels
        enriched = []
        for edge in edges:
            src = self.graph.get_node(edge.source_id)
            tgt = self.graph.get_node(edge.target_id)
            enriched.append({
                "source": src.label if src else edge.source_id,
                "target": tgt.label if tgt else edge.target_id,
                "type": edge.edge_type,
                "weight": edge.weight,
            })

        return QueryResult(query, "edges", enriched,
                           f"Relations of type '{rel_type}'")

    def _handle_entity_type(self, query: str) -> QueryResult:
        """Handle 'entities of type X [who are connected to Y]' queries."""
        match = re.search(
            r'entities\s+of\s+type\s+(\w+)'
            r'(?:\s+(?:who\s+are\s+)?connected\s+to\s+(\w+))?',
            query, re.IGNORECASE
        )
        if not match:
            return QueryResult(query, "nodes", [], "Could not parse entity type query.")

        entity_type = match.group(1).upper()
        connected_type = match.group(2)

        nodes = [n for n in self.graph.nodes.values()
                 if n.node_type == entity_type]

        if connected_type:
            connected_type = connected_type.upper()
            filtered = []
            for node in nodes:
                neighbor_ids = self.graph.get_neighbors(node.node_id)
                for nid in neighbor_ids:
                    neighbor = self.graph.get_node(nid)
                    if neighbor and neighbor.node_type == connected_type:
                        filtered.append(node)
                        break
            nodes = filtered

        desc = f"Entities of type {entity_type}"
        if connected_type:
            desc += f" connected to {connected_type}"

        return QueryResult(query, "nodes", nodes, desc)

    def _handle_info(self, query: str) -> QueryResult:
        """Handle 'info about X' queries."""
        match = re.search(r'(?:info|details)\s+(?:about|of)\s+(.+?)$',
                          query, re.IGNORECASE)
        if not match:
            return QueryResult(query, "stats", {}, "Could not parse info query.")

        entity_name = match.group(1).strip().strip('"\'')
        node_id = self.graph._generate_id(entity_name)

        if node_id not in self.graph.nodes:
            node_id = self._fuzzy_find_node(entity_name)

        if not node_id:
            return QueryResult(query, "stats", {},
                               f"Entity '{entity_name}' not found.")

        node = self.graph.nodes[node_id]
        outgoing = self.graph.get_edges(source_id=node_id)
        incoming = self.graph.get_edges(target_id=node_id)

        info = {
            "label": node.label,
            "type": node.node_type,
            "properties": node.properties,
            "degree": self.graph.get_degree(node_id),
            "outgoing_relations": len(outgoing),
            "incoming_relations": len(incoming),
            "outgoing": [
                f"--[{e.edge_type}]--> {self.graph.nodes[e.target_id].label}"
                for e in outgoing if e.target_id in self.graph.nodes
            ],
            "incoming": [
                f"<--[{e.edge_type}]-- {self.graph.nodes[e.source_id].label}"
                for e in incoming if e.source_id in self.graph.nodes
            ],
        }

        return QueryResult(query, "stats", info,
                           f"Information about '{node.label}'")

    def _handle_count(self, query: str) -> QueryResult:
        """Handle count queries."""
        lower = query.lower()
        if "node" in lower or "entit" in lower:
            return QueryResult(query, "stats",
                               {"total_entities": len(self.graph.nodes)})
        elif "edge" in lower or "relation" in lower:
            return QueryResult(query, "stats",
                               {"total_relations": len(self.graph.edges)})
        else:
            return QueryResult(query, "stats", {
                "total_entities": len(self.graph.nodes),
                "total_relations": len(self.graph.edges),
            })

    def _handle_stats(self, query: str) -> QueryResult:
        """Handle stats/statistics queries."""
        stats = self.graph.get_statistics()
        return QueryResult(query, "stats", stats, "Graph statistics")

    def _handle_connected_to(self, query: str) -> QueryResult:
        """Handle 'connected to X' queries."""
        match = re.search(r'connected\s+to\s+(.+?)$', query, re.IGNORECASE)
        if not match:
            return QueryResult(query, "nodes", [], "Could not parse query.")

        entity_name = match.group(1).strip().strip('"\'')
        return self._find_connected(entity_name)

    def _handle_general_search(self, query: str) -> QueryResult:
        """Handle general text search across node labels."""
        results = []
        query_lower = query.lower().strip('"\'')

        for node in self.graph.nodes.values():
            if query_lower in node.label.lower():
                results.append(node)

        if results:
            return QueryResult(query, "nodes", results,
                               f"Search results for '{query}'")
        else:
            return QueryResult(query, "nodes", [],
                               f"No results found for '{query}'")

    def _find_connected(self, entity_name: str,
                        filter_type: Optional[str] = None) -> QueryResult:
        """Find all entities connected to a given entity."""
        node_id = self.graph._generate_id(entity_name)

        if node_id not in self.graph.nodes:
            node_id = self._fuzzy_find_node(entity_name)

        if not node_id:
            return QueryResult(f"connected to {entity_name}", "nodes", [],
                               f"Entity '{entity_name}' not found.")

        neighbor_ids = self.graph.get_neighbors(node_id)
        nodes = [self.graph.nodes[nid] for nid in neighbor_ids
                 if nid in self.graph.nodes]

        if filter_type:
            filter_type = filter_type.upper()
            nodes = [n for n in nodes if n.node_type == filter_type]

        return QueryResult(f"connected to {entity_name}", "nodes", nodes,
                           f"Entities connected to '{entity_name}'")

    def _find_by_type(self, entity_type: str,
                      connected_type: Optional[str] = None) -> QueryResult:
        """Find entities by type, optionally filtered by connection type."""
        nodes = [n for n in self.graph.nodes.values()
                 if n.node_type == entity_type]

        if connected_type:
            connected_type = connected_type.upper()
            filtered = []
            for node in nodes:
                neighbor_ids = self.graph.get_neighbors(node.node_id)
                for nid in neighbor_ids:
                    neighbor = self.graph.get_node(nid)
                    if neighbor and neighbor.node_type == connected_type:
                        filtered.append(node)
                        break
            nodes = filtered

        return QueryResult(f"type {entity_type}", "nodes", nodes,
                           f"Entities of type {entity_type}")

    def _fuzzy_find_node(self, name: str) -> Optional[str]:
        """Attempt fuzzy matching on node labels."""
        name_lower = name.lower()

        # Exact match on label
        for nid, node in self.graph.nodes.items():
            if node.label.lower() == name_lower:
                return nid

        # Substring match
        for nid, node in self.graph.nodes.items():
            if name_lower in node.label.lower() or node.label.lower() in name_lower:
                return nid

        return None

    def interactive_help(self) -> str:
        """Return help text for query syntax."""
        return """
Knowledge Graph Query Language
==============================

Supported query patterns:

  find all entities connected to <entity>
  find all PERSON connected to ORGANIZATION
  path between <entity1> and <entity2>
  neighbors of <entity>
  relations of type <relation_type>
  entities of type <PERSON|ORGANIZATION|LOCATION|DATE|TECHNOLOGY>
  entities of type PERSON connected to ORGANIZATION
  info about <entity>
  count nodes
  count edges
  stats

Relation types: founded_by, located_in, works_for, acquired,
                developed, born_in, partner_of, competes_with

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, TECHNOLOGY

Examples:
  find all entities connected to Google
  path between Elon Musk and Apple
  entities of type PERSON who are connected to ORGANIZATION
  relations of type founded_by
  info about Tesla
  neighbors of Microsoft
"""
