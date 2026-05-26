"""
Relationship extraction using verb-based dependency-like patterns
and co-occurrence analysis.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from .entity_extractor import Entity


class Relation:
    """Represents an extracted relationship between two entities."""

    def __init__(self, source: Entity, target: Entity, relation_type: str,
                 confidence: float = 1.0, evidence: str = ""):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.confidence = confidence
        self.evidence = evidence

    def __repr__(self):
        return (f"Relation({self.source.text!r} --[{self.relation_type}]--> "
                f"{self.target.text!r}, conf={self.confidence:.2f})")

    def __eq__(self, other):
        if isinstance(other, Relation):
            return (self.source.text == other.source.text and
                    self.target.text == other.target.text and
                    self.relation_type == other.relation_type)
        return False

    def __hash__(self):
        return hash((self.source.text, self.target.text, self.relation_type))


# ── Verb patterns that indicate relationships ───────────────────────────────

FOUNDED_PATTERNS = [
    r'{E1}\s+(?:co-)?founded\s+{E2}',
    r'{E2}\s+was\s+(?:co-)?founded\s+by\s+{E1}',
    r'{E1}\s+(?:co-)?established\s+{E2}',
    r'{E2}\s+was\s+(?:co-)?established\s+by\s+{E1}',
    r'{E1}\s+(?:co-)?created\s+{E2}',
    r'{E1}\s+(?:and\s+\w+\s+)?(?:co-)?founded\s+{E2}',
    r'{E1},?\s+(?:the\s+)?(?:co-)?founder\s+of\s+{E2}',
    r'{E1},?\s+who\s+(?:co-)?founded\s+{E2}',
]

LOCATED_IN_PATTERNS = [
    r'{E1}\s+(?:is\s+)?(?:located|based|headquartered|situated)\s+in\s+{E2}',
    r'{E1}\s+in\s+{E2}',
    r'{E1},?\s+{E2}',  # "Google, Mountain View" pattern
    r'{E2}-based\s+{E1}',
    r'{E1}\s+has\s+(?:its\s+)?headquarters\s+in\s+{E2}',
    r'{E1}\s+moved\s+(?:its\s+headquarters\s+)?to\s+{E2}',
    r'{E1}\s+offices?\s+in\s+{E2}',
]

WORKS_FOR_PATTERNS = [
    r'{E1}\s+(?:works|worked)\s+(?:for|at)\s+{E2}',
    r'{E1},?\s+(?:a|the)?\s*(?:CEO|CTO|CFO|COO|president|chairman|director|engineer|scientist|researcher|executive|officer|head|chief|VP|vice president|manager|lead|principal)\s+(?:of|at)\s+{E2}',
    r'{E1}\s+(?:joined|left|leads?|led|manages?|managed|runs?|ran|heads?|headed)\s+{E2}',
    r'{E1}\s+(?:serves?|served)\s+as\s+\w+\s+(?:of|at)\s+{E2}',
    r'{E2}\s+(?:hired|appointed|named)\s+{E1}',
    r'{E1}\s+at\s+{E2}',
]

ACQUIRED_PATTERNS = [
    r'{E1}\s+acquired\s+{E2}',
    r'{E2}\s+was\s+acquired\s+by\s+{E1}',
    r'{E1}\s+(?:bought|purchased)\s+{E2}',
    r'{E1}\s+merged\s+with\s+{E2}',
    r'{E1}\s+took\s+over\s+{E2}',
    r'{E1}\'s?\s+acquisition\s+of\s+{E2}',
]

DEVELOPED_PATTERNS = [
    r'{E1}\s+(?:developed|created|built|designed|invented|introduced|launched|released)\s+{E2}',
    r'{E2}\s+was\s+(?:developed|created|built|designed|invented|introduced|launched|released)\s+by\s+{E1}',
    r'{E1}\s+(?:pioneered|engineered|produced)\s+{E2}',
    r'{E1}\'s?\s+{E2}',
]

BORN_IN_PATTERNS = [
    r'{E1}\s+was\s+born\s+in\s+{E2}',
    r'{E1},?\s+born\s+in\s+{E2}',
    r'{E1}\s+(?:grew\s+up|raised)\s+in\s+{E2}',
    r'{E1}\s+(?:is\s+)?from\s+{E2}',
    r'{E1},?\s+(?:a|an)\s+\w+\s+from\s+{E2}',
]

PARTNER_PATTERNS = [
    r'{E1}\s+partnered\s+with\s+{E2}',
    r'{E1}\s+(?:and|&)\s+{E2}\s+(?:partnered|collaborated|teamed\s+up|joined\s+forces)',
    r'partnership\s+between\s+{E1}\s+and\s+{E2}',
    r'{E1}\s+collaborated\s+with\s+{E2}',
    r'{E1}\s+(?:and|&)\s+{E2}\s+(?:announced|formed|signed)\s+(?:a\s+)?(?:partnership|alliance|deal|agreement)',
    r'{E1}\s+invested\s+in\s+{E2}',
]

COMPETES_WITH_PATTERNS = [
    r'{E1}\s+competes?\s+with\s+{E2}',
    r'{E1}\s+(?:and|&)\s+{E2}\s+(?:are|were)\s+(?:rivals?|competitors?)',
    r'rivalry\s+between\s+{E1}\s+and\s+{E2}',
    r'{E1}\s+(?:rivaled?|challenged?)\s+{E2}',
    r'{E1}\s+(?:vs\.?|versus)\s+{E2}',
]

# Map relation types to their patterns and valid entity type pairs
RELATION_DEFINITIONS = {
    "founded_by": {
        "patterns": FOUNDED_PATTERNS,
        "valid_pairs": [("PERSON", "ORGANIZATION")],
        "reverse": False,  # E1->E2 direction
    },
    "located_in": {
        "patterns": LOCATED_IN_PATTERNS,
        "valid_pairs": [
            ("ORGANIZATION", "LOCATION"),
            ("PERSON", "LOCATION"),
        ],
        "reverse": False,
    },
    "works_for": {
        "patterns": WORKS_FOR_PATTERNS,
        "valid_pairs": [("PERSON", "ORGANIZATION")],
        "reverse": False,
    },
    "acquired": {
        "patterns": ACQUIRED_PATTERNS,
        "valid_pairs": [("ORGANIZATION", "ORGANIZATION")],
        "reverse": False,
    },
    "developed": {
        "patterns": DEVELOPED_PATTERNS,
        "valid_pairs": [
            ("ORGANIZATION", "TECHNOLOGY"),
            ("PERSON", "TECHNOLOGY"),
        ],
        "reverse": False,
    },
    "born_in": {
        "patterns": BORN_IN_PATTERNS,
        "valid_pairs": [("PERSON", "LOCATION"), ("PERSON", "DATE")],
        "reverse": False,
    },
    "partner_of": {
        "patterns": PARTNER_PATTERNS,
        "valid_pairs": [
            ("ORGANIZATION", "ORGANIZATION"),
            ("PERSON", "ORGANIZATION"),
        ],
        "reverse": False,
    },
    "competes_with": {
        "patterns": COMPETES_WITH_PATTERNS,
        "valid_pairs": [("ORGANIZATION", "ORGANIZATION")],
        "reverse": False,
    },
}


class RelationExtractor:
    """Extract relationships between entities from text."""

    def __init__(self):
        self.relations: List[Relation] = []

    def extract(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relationships from text given a list of entities."""
        relations = []

        # Build entity lookup by text for quick access
        entity_map: Dict[str, Entity] = {}
        for ent in entities:
            entity_map[ent.text] = ent

        # Split into sentences for context
        sentences = self._split_sentences(text)

        for sentence in sentences:
            # Find which entities appear in this sentence
            sent_entities = self._find_entities_in_sentence(sentence, entities)

            if len(sent_entities) < 2:
                continue

            # Try pattern-based extraction for each entity pair
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i + 1:]:
                    rels = self._extract_pattern_relations(sentence, ent1, ent2)
                    relations.extend(rels)

            # Co-occurrence based relations (lower confidence)
            cooc_rels = self._extract_cooccurrence_relations(sentence, sent_entities)
            relations.extend(cooc_rels)

        # Deduplicate
        self.relations = self._deduplicate(relations)
        return self.relations

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        text = re.sub(r'\s+', ' ', text.strip())
        # Split on period, exclamation, question mark followed by space and capital
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def _find_entities_in_sentence(self, sentence: str, entities: List[Entity]) -> List[Entity]:
        """Find which entities appear in a given sentence."""
        found = []
        for ent in entities:
            if ent.text in sentence:
                found.append(ent)
        return found

    def _extract_pattern_relations(self, sentence: str, ent1: Entity,
                                   ent2: Entity) -> List[Relation]:
        """Try to match verb-based patterns between two entities."""
        relations = []

        for rel_type, definition in RELATION_DEFINITIONS.items():
            valid_pairs = definition["valid_pairs"]
            patterns = definition["patterns"]

            # Check both orderings of entity pair
            for pair in valid_pairs:
                src_type, tgt_type = pair

                # Try E1=ent1, E2=ent2
                if ent1.entity_type == src_type and ent2.entity_type == tgt_type:
                    if self._match_patterns(sentence, ent1.text, ent2.text, patterns):
                        rel = Relation(ent1, ent2, rel_type,
                                       confidence=0.85, evidence=sentence)
                        relations.append(rel)

                # Try E1=ent2, E2=ent1
                if ent2.entity_type == src_type and ent1.entity_type == tgt_type:
                    if self._match_patterns(sentence, ent2.text, ent1.text, patterns):
                        rel = Relation(ent2, ent1, rel_type,
                                       confidence=0.85, evidence=sentence)
                        relations.append(rel)

        return relations

    def _match_patterns(self, sentence: str, e1_text: str, e2_text: str,
                        patterns: List[str]) -> bool:
        """Check if any pattern matches in the sentence."""
        for pattern_template in patterns:
            pattern = pattern_template.replace('{E1}', re.escape(e1_text))
            pattern = pattern.replace('{E2}', re.escape(e2_text))
            try:
                if re.search(pattern, sentence, re.IGNORECASE):
                    return True
            except re.error:
                continue
        return False

    def _extract_cooccurrence_relations(self, sentence: str,
                                        sent_entities: List[Entity]) -> List[Relation]:
        """
        Extract relations based on co-occurrence with lower confidence.
        Used when no explicit pattern matches but entities appear close together.
        """
        relations = []

        for i, ent1 in enumerate(sent_entities):
            for ent2 in sent_entities[i + 1:]:
                # Skip if already found by pattern matching
                if self._already_related(ent1, ent2):
                    continue

                # Check proximity (entities within 10 words of each other)
                distance = self._entity_distance(sentence, ent1.text, ent2.text)
                if distance is None or distance > 10:
                    continue

                # Infer relation type from entity types
                rel_type = self._infer_relation_type(ent1, ent2, sentence)
                if rel_type:
                    confidence = max(0.3, 0.6 - (distance * 0.03))
                    rel = Relation(ent1, ent2, rel_type,
                                   confidence=confidence, evidence=sentence)
                    relations.append(rel)

        return relations

    def _already_related(self, ent1: Entity, ent2: Entity) -> bool:
        """Check if two entities are already related."""
        for rel in self.relations:
            if ((rel.source.text == ent1.text and rel.target.text == ent2.text) or
                (rel.source.text == ent2.text and rel.target.text == ent1.text)):
                return True
        return False

    def _entity_distance(self, sentence: str, e1: str, e2: str) -> Optional[int]:
        """Count words between two entity mentions in a sentence."""
        idx1 = sentence.find(e1)
        idx2 = sentence.find(e2)
        if idx1 == -1 or idx2 == -1:
            return None

        if idx1 < idx2:
            between = sentence[idx1 + len(e1):idx2]
        else:
            between = sentence[idx2 + len(e2):idx1]

        words = between.split()
        return len(words)

    def _infer_relation_type(self, ent1: Entity, ent2: Entity,
                             sentence: str) -> Optional[str]:
        """Infer a likely relation type from entity type pair and context."""
        pair = (ent1.entity_type, ent2.entity_type)
        sent_lower = sentence.lower()

        type_map = {
            ("PERSON", "ORGANIZATION"): self._infer_person_org(sent_lower),
            ("ORGANIZATION", "PERSON"): self._infer_person_org(sent_lower, reverse=True),
            ("PERSON", "LOCATION"): "born_in",
            ("ORGANIZATION", "LOCATION"): "located_in",
            ("LOCATION", "ORGANIZATION"): "located_in",
            ("ORGANIZATION", "ORGANIZATION"): self._infer_org_org(sent_lower),
            ("PERSON", "TECHNOLOGY"): "developed",
            ("ORGANIZATION", "TECHNOLOGY"): "developed",
            ("PERSON", "DATE"): "born_in",
        }

        result = type_map.get(pair)
        # Handle swapped source/target for certain types
        if pair == ("LOCATION", "ORGANIZATION"):
            # Swap: org located_in location
            return None  # We handle this with the reversed pair
        return result

    def _infer_person_org(self, context: str, reverse: bool = False) -> str:
        """Infer whether a person-org relation is works_for or founded_by."""
        if any(w in context for w in ["founded", "co-founded", "established", "created", "started"]):
            return "founded_by" if not reverse else "founded_by"
        return "works_for"

    def _infer_org_org(self, context: str) -> str:
        """Infer org-org relation type."""
        if any(w in context for w in ["acquired", "bought", "purchased", "merged", "takeover"]):
            return "acquired"
        if any(w in context for w in ["partner", "collaborated", "alliance", "deal", "invested"]):
            return "partner_of"
        if any(w in context for w in ["compet", "rival", "versus", "vs"]):
            return "competes_with"
        return "partner_of"

    def _deduplicate(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations, keeping the highest confidence."""
        seen: Dict[Tuple[str, str, str], Relation] = {}

        for rel in relations:
            key = (rel.source.text, rel.target.text, rel.relation_type)
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel

        return list(seen.values())

    def get_relations_by_type(self, relation_type: str) -> List[Relation]:
        """Get all relations of a specific type."""
        return [r for r in self.relations if r.relation_type == relation_type]

    def get_relations_for_entity(self, entity_text: str) -> List[Relation]:
        """Get all relations involving a specific entity."""
        return [r for r in self.relations
                if r.source.text == entity_text or r.target.text == entity_text]

    def get_relation_summary(self) -> Dict[str, int]:
        """Get counts of relations by type."""
        summary = {}
        for rel in self.relations:
            summary[rel.relation_type] = summary.get(rel.relation_type, 0) + 1
        return summary
