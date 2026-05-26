# Knowledge Graph Builder & Explorer

A Python system that extracts entities and relationships from text to build knowledge graphs, then provides graph analysis and visualization. No external NLP libraries required -- all extraction is pattern and rule-based.

## Features

- **Entity Extraction**: Identifies PERSON, ORGANIZATION, LOCATION, DATE, and TECHNOLOGY entities using gazetteers, pattern matching, and contextual heuristics
- **Relationship Extraction**: Extracts relationships (founded_by, located_in, works_for, acquired, developed, born_in, partner_of, competes_with) using verb-based patterns and co-occurrence analysis
- **Knowledge Graph**: Typed nodes and edges with CRUD operations, JSON serialization, and entity deduplication
- **Graph Algorithms**: PageRank, betweenness/degree/closeness centrality, community detection (Louvain), shortest path, connected components
- **Query Engine**: Natural-ish query language for graph traversal
- **Visualization**: Force-directed layout from scratch with matplotlib rendering
- **Built-in Corpus**: 25+ paragraphs covering tech companies, AI, computing history, science, and more

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Output

Running `main.py` produces:
- `knowledge_graph.json` -- the full graph in JSON format
- `knowledge_graph.png` -- main graph visualization
- `communities.png` -- community structure visualization
- `subgraph_google.png` -- subgraph around Google

## Project Structure

```
knowledge_graph_explorer/
    __init__.py
    tokenizer.py           - Sentence/word tokenizer, POS-like tagging
    entity_extractor.py    - Named entity recognition (pattern-based)
    relation_extractor.py  - Relationship extraction (verb patterns)
    knowledge_graph.py     - Graph data structure, CRUD, serialization
    graph_algorithms.py    - PageRank, communities, shortest path, centrality
    query_engine.py        - Query language for graph traversal
    visualization.py       - Force-directed layout & matplotlib rendering
    datasets.py            - Built-in text corpus (25+ paragraphs)
    main.py                - Pipeline: extract, build, analyze, visualize
```

## Query Examples

```python
from knowledge_graph_explorer.query_engine import QueryEngine

qe = QueryEngine(graph)
qe.execute("find all entities connected to Google")
qe.execute("path between Elon Musk and Apple")
qe.execute("entities of type PERSON connected to ORGANIZATION")
qe.execute("relations of type founded_by")
qe.execute("info about Tesla")
```

## Requirements

- Python 3.8+
- numpy
- matplotlib
