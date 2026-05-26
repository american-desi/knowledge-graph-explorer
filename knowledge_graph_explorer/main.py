#!/usr/bin/env python3
"""
Knowledge Graph Builder & Explorer
===================================
Process text corpus, extract entities and relationships,
build a knowledge graph, run analyses, and visualize.
"""

import os
import sys
import json
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph_explorer.datasets import get_all_texts, get_corpus_stats
from knowledge_graph_explorer.entity_extractor import EntityExtractor
from knowledge_graph_explorer.relation_extractor import RelationExtractor
from knowledge_graph_explorer.knowledge_graph import KnowledgeGraph
from knowledge_graph_explorer.graph_algorithms import (
    pagerank, betweenness_centrality, degree_centrality,
    connected_components, community_detection_louvain,
    shortest_path, get_comprehensive_statistics, modularity,
)
from knowledge_graph_explorer.query_engine import QueryEngine
from knowledge_graph_explorer.visualization import (
    visualize_graph, visualize_communities, visualize_subgraph,
)


def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_subheader(title: str):
    """Print a formatted sub-section header."""
    print(f"\n  --- {title} ---")


def main():
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Step 1: Load Corpus ───────────────────────────────────────────────
    print_header("KNOWLEDGE GRAPH BUILDER & EXPLORER")
    print_header("Step 1: Loading Text Corpus", "-")

    texts = get_all_texts()
    corpus_stats = get_corpus_stats()

    print(f"  Loaded {corpus_stats['num_paragraphs']} paragraphs")
    print(f"  Total words: {corpus_stats['total_words']}")
    print(f"  Avg words/paragraph: {corpus_stats['avg_words_per_paragraph']:.1f}")

    # ── Step 2: Extract Entities ──────────────────────────────────────────
    print_header("Step 2: Extracting Entities", "-")

    extractor = EntityExtractor()
    combined_text = "\n\n".join(texts)
    entities = extractor.extract(combined_text)

    summary = extractor.get_entity_summary()
    print(f"  Extracted {len(entities)} unique entities:")
    for etype, count in sorted(summary.items()):
        print(f"    {etype}: {count}")

    print_subheader("Sample Entities by Type")
    for etype in ["PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "DATE"]:
        type_entities = extractor.get_entities_by_type(etype)
        if type_entities:
            samples = [e.text for e in sorted(type_entities,
                                               key=lambda x: x.confidence,
                                               reverse=True)[:8]]
            print(f"  {etype}: {', '.join(samples)}")

    # ── Step 3: Extract Relations ─────────────────────────────────────────
    print_header("Step 3: Extracting Relationships", "-")

    rel_extractor = RelationExtractor()
    relations = rel_extractor.extract(combined_text, entities)

    rel_summary = rel_extractor.get_relation_summary()
    print(f"  Extracted {len(relations)} relationships:")
    for rtype, count in sorted(rel_summary.items()):
        print(f"    {rtype}: {count}")

    print_subheader("Sample Relationships")
    for rtype in ["founded_by", "works_for", "located_in", "acquired",
                  "developed", "competes_with", "partner_of"]:
        type_rels = rel_extractor.get_relations_by_type(rtype)
        if type_rels:
            print(f"\n  {rtype}:")
            for rel in type_rels[:5]:
                print(f"    {rel.source.text} --> {rel.target.text}"
                      f"  (conf: {rel.confidence:.2f})")

    # ── Step 4: Build Knowledge Graph ─────────────────────────────────────
    print_header("Step 4: Building Knowledge Graph", "-")

    kg = KnowledgeGraph()
    kg.build_from_extractions(entities, relations)

    print(f"  Graph: {kg}")
    stats = kg.get_statistics()
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    print(f"  Max degree: {stats['max_degree']}")
    print(f"  Isolated nodes: {stats['isolated_nodes']}")

    print_subheader("Node Types")
    for ntype, count in sorted(stats['node_types'].items()):
        print(f"    {ntype}: {count}")

    print_subheader("Edge Types")
    for etype, count in sorted(stats['edge_types'].items()):
        print(f"    {etype}: {count}")

    # ── Step 5: Graph Analysis ────────────────────────────────────────────
    print_header("Step 5: Graph Analysis", "-")

    # PageRank
    print_subheader("Top 15 Entities by PageRank")
    pr = pagerank(kg)
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:15]
    for rank, (nid, score) in enumerate(top_pr, 1):
        node = kg.get_node(nid)
        if node:
            print(f"    {rank:2d}. {node.label:<30s} [{node.node_type:<12s}]  "
                  f"score: {score:.4f}")

    # Betweenness Centrality
    print_subheader("Top 10 Entities by Betweenness Centrality")
    bc = betweenness_centrality(kg)
    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (nid, score) in enumerate(top_bc, 1):
        node = kg.get_node(nid)
        if node:
            print(f"    {rank:2d}. {node.label:<30s} [{node.node_type:<12s}]  "
                  f"score: {score:.4f}")

    # Degree Centrality
    print_subheader("Top 10 Entities by Degree Centrality")
    dc = degree_centrality(kg)
    top_dc = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (nid, score) in enumerate(top_dc, 1):
        node = kg.get_node(nid)
        if node:
            print(f"    {rank:2d}. {node.label:<30s} [{node.node_type:<12s}]  "
                  f"score: {score:.4f}")

    # Connected Components
    print_subheader("Connected Components")
    components = connected_components(kg)
    print(f"    Number of components: {len(components)}")
    for i, comp in enumerate(components[:5]):
        labels = [kg.get_node(nid).label for nid in list(comp)[:8]
                  if kg.get_node(nid)]
        suffix = "..." if len(comp) > 8 else ""
        print(f"    Component {i} (size {len(comp)}): "
              f"{', '.join(labels)}{suffix}")

    # Community Detection
    print_subheader("Community Detection (Louvain)")
    communities = community_detection_louvain(kg)
    num_communities = len(set(communities.values()))
    mod = modularity(kg, communities)
    print(f"    Detected {num_communities} communities")
    print(f"    Modularity: {mod:.4f}")

    comm_members = defaultdict(list)
    for nid, comm in communities.items():
        node = kg.get_node(nid)
        if node:
            comm_members[comm].append(node.label)

    for comm_id in sorted(comm_members.keys()):
        members = comm_members[comm_id]
        display = members[:8]
        suffix = f"... (+{len(members) - 8} more)" if len(members) > 8 else ""
        print(f"    Community {comm_id} ({len(members)} members): "
              f"{', '.join(display)}{suffix}")

    # Shortest Paths
    print_subheader("Interesting Paths")
    path_queries = [
        ("Elon Musk", "Google"),
        ("Alan Turing", "Python"),
        ("Bill Gates", "Amazon"),
        ("Tim Berners-Lee", "ChatGPT"),
        ("Steve Jobs", "Samsung"),
    ]
    for source, target in path_queries:
        path = shortest_path(kg, source, target)
        if path:
            labels = [kg.get_node(nid).label if kg.get_node(nid) else nid
                      for nid in path]
            print(f"    {source} -> {target}: {' -> '.join(labels)}")
        else:
            print(f"    {source} -> {target}: No path found")

    # ── Step 6: Query Engine Demo ─────────────────────────────────────────
    print_header("Step 6: Query Engine Demo", "-")

    qe = QueryEngine(kg)

    queries = [
        "find all entities connected to Google",
        "path between Elon Musk and Apple",
        "entities of type PERSON connected to ORGANIZATION",
        "relations of type founded_by",
        "relations of type acquired",
        "neighbors of Microsoft",
        "info about Tesla",
        "count nodes",
        "stats",
    ]

    for query_str in queries:
        result = qe.execute(query_str)
        print(f"\n  > {query_str}")
        # Print abbreviated results
        if result.result_type == "nodes":
            items = result.data[:6]
            for item in items:
                if hasattr(item, 'label'):
                    print(f"      {item.label} [{item.node_type}]")
                elif isinstance(item, dict):
                    print(f"      {item.get('label', '?')} [{item.get('type', '?')}]")
            if len(result.data) > 6:
                print(f"      ... and {len(result.data) - 6} more")
        elif result.result_type == "edges":
            items = result.data[:6]
            for item in items:
                if isinstance(item, dict):
                    print(f"      {item['source']} --[{item['type']}]--> {item['target']}")
                else:
                    print(f"      {item.source_id} --[{item.edge_type}]--> {item.target_id}")
            if len(result.data) > 6:
                print(f"      ... and {len(result.data) - 6} more")
        elif result.result_type == "path":
            if result.data:
                print(f"      Path: {' -> '.join(result.data)}")
            else:
                print(f"      No path found.")
        elif result.result_type == "stats":
            for k, v in list(result.data.items())[:10]:
                print(f"      {k}: {v}")

    # ── Step 7: Export Graph ──────────────────────────────────────────────
    print_header("Step 7: Exporting Graph", "-")

    json_path = os.path.join(output_dir, "knowledge_graph.json")
    kg.to_json(json_path)
    print(f"  Graph exported to: {json_path}")

    # Verify round-trip
    loaded = KnowledgeGraph.from_json(json_path)
    print(f"  Verified import: {loaded}")

    # ── Step 8: Visualization ─────────────────────────────────────────────
    print_header("Step 8: Generating Visualizations", "-")

    # Main graph visualization
    main_viz_path = os.path.join(output_dir, "knowledge_graph.png")
    print("\n  [1/3] Main knowledge graph:")
    result = visualize_graph(
        kg,
        output_path=main_viz_path,
        title="Knowledge Graph: Tech Industry & Science",
        figsize=(24, 18),
        layout_iterations=250,
        font_size=7,
        dpi=150,
    )

    # Community visualization
    comm_viz_path = os.path.join(output_dir, "communities.png")
    print("\n  [2/3] Community structure:")
    visualize_communities(
        kg, communities,
        output_path=comm_viz_path,
        title="Knowledge Graph Communities",
        figsize=(24, 18),
        dpi=150,
    )

    # Subgraph visualization
    sub_viz_path = os.path.join(output_dir, "subgraph_google.png")
    print("\n  [3/3] Subgraph around Google:")
    visualize_subgraph(
        kg,
        center_label="Google",
        max_depth=2,
        output_path=sub_viz_path,
        figsize=(18, 14),
        dpi=150,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print_header("SUMMARY")
    print(f"  Corpus: {corpus_stats['num_paragraphs']} paragraphs, "
          f"{corpus_stats['total_words']} words")
    print(f"  Entities extracted: {len(entities)}")
    print(f"  Relationships extracted: {len(relations)}")
    print(f"  Graph nodes: {len(kg.nodes)}")
    print(f"  Graph edges: {len(kg.edges)}")
    print(f"  Connected components: {len(components)}")
    print(f"  Communities detected: {num_communities}")
    print(f"  Modularity: {mod:.4f}")
    print(f"\n  Output files:")
    print(f"    - {json_path}")
    print(f"    - {main_viz_path}")
    print(f"    - {comm_viz_path}")
    print(f"    - {sub_viz_path}")
    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
