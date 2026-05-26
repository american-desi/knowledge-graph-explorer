"""
Simple but effective sentence/word tokenizer with POS-like tagging using patterns.
No external NLP libraries required.
"""

import re
from typing import List, Tuple


# Common word lists for POS-like tagging
DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "no", "every", "each", "all",
    "both", "few", "several", "many", "much", "most",
}

PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "near", "behind", "across", "along",
    "around", "among", "within", "without", "upon", "against", "throughout",
}

CONJUNCTIONS = {
    "and", "or", "but", "nor", "for", "yet", "so", "while", "although",
    "because", "since", "unless", "if", "when", "where", "whereas",
}

PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "that", "what",
}

AUXILIARY_VERBS = {
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
}

COMMON_VERBS = {
    "founded", "co-founded", "established", "created", "built", "developed",
    "acquired", "bought", "purchased", "merged",
    "works", "worked", "working",
    "located", "based", "headquartered", "situated",
    "born", "died", "lived",
    "launched", "released", "introduced", "announced", "unveiled",
    "invested", "funded", "raised", "valued",
    "partnered", "collaborated", "joined", "hired", "appointed",
    "leads", "led", "manages", "managed", "runs", "ran",
    "invented", "discovered", "pioneered", "designed", "engineered",
    "competes", "competed", "rivaled", "challenged",
    "said", "stated", "announced", "reported", "revealed",
    "moved", "relocated", "expanded", "opened", "closed",
    "became", "serves", "served", "known", "named", "called",
    "operates", "operated", "produces", "produced", "manufactures",
    "sold", "sells", "offered", "provides", "provided",
    "includes", "included", "contains", "contained",
    "grew", "grows", "increased", "decreased", "rose", "fell",
    "won", "received", "awarded", "earned", "gained",
    "published", "wrote", "authored", "contributed",
    "studied", "researched", "explored", "investigated",
    "transformed", "changed", "revolutionized", "disrupted",
    "connected", "linked", "associated", "related",
}

COMMON_ADJECTIVES = {
    "large", "small", "big", "new", "old", "young", "major", "minor",
    "first", "second", "third", "last", "next", "previous",
    "global", "international", "national", "local", "regional",
    "public", "private", "open", "closed",
    "leading", "top", "best", "worst", "largest", "smallest",
    "important", "significant", "key", "main", "primary",
    "american", "british", "chinese", "european", "japanese",
    "early", "late", "modern", "ancient", "recent",
    "popular", "famous", "well-known", "renowned",
    "successful", "powerful", "influential", "dominant",
    "digital", "electronic", "technical", "scientific",
    "social", "economic", "political", "cultural",
}

COMMON_ADVERBS = {
    "also", "later", "then", "now", "today", "currently", "recently",
    "originally", "previously", "formerly", "initially", "eventually",
    "subsequently", "ultimately", "finally", "already", "soon",
    "very", "really", "quite", "rather", "especially", "particularly",
    "not", "never", "always", "often", "sometimes", "usually",
    "here", "there", "where", "everywhere", "anywhere",
    "well", "quickly", "slowly", "rapidly", "significantly",
    "approximately", "nearly", "almost", "about", "roughly",
    "together", "simultaneously", "jointly", "independently",
}


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Handle common abbreviations that shouldn't end sentences
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.",
        "Inc.", "Ltd.", "Corp.", "Co.", "vs.", "etc.", "e.g.",
        "i.e.", "U.S.", "U.K.", "E.U.", "St.", "Ave.", "Blvd.",
    ]
    placeholder_map = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholder_map[placeholder] = abbr
        text = text.replace(abbr, placeholder)

    # Split on sentence-ending punctuation followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore abbreviations
    restored = []
    for sent in sentences:
        for placeholder, abbr in placeholder_map.items():
            sent = sent.replace(placeholder, abbr)
        sent = sent.strip()
        if sent:
            restored.append(sent)

    return restored


def tokenize_words(sentence: str) -> List[str]:
    """Split a sentence into word tokens."""
    # Handle contractions and possessives
    tokens = re.findall(r"[\w]+(?:'[\w]+)?|[.,;:!?\"'()\-/]", sentence)
    return [t for t in tokens if t.strip()]


def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Assign POS-like tags to tokens using pattern matching and word lists.

    Tags:
        DET - Determiner
        PREP - Preposition
        CONJ - Conjunction
        PRON - Pronoun
        AUX - Auxiliary verb
        VERB - Verb
        ADJ - Adjective
        ADV - Adverb
        NUM - Number
        PROPN - Proper noun (capitalized)
        NOUN - Common noun
        PUNCT - Punctuation
    """
    tagged = []
    for i, token in enumerate(tokens):
        lower = token.lower()

        # Punctuation
        if re.match(r'^[.,;:!?"\'()\-/]+$', token):
            tagged.append((token, "PUNCT"))

        # Numbers and dates
        elif re.match(r'^\d+$', token):
            tagged.append((token, "NUM"))
        elif re.match(r'^\d{4}s?$', token):
            tagged.append((token, "DATE"))

        # Word-list lookups
        elif lower in DETERMINERS:
            tagged.append((token, "DET"))
        elif lower in PREPOSITIONS:
            tagged.append((token, "PREP"))
        elif lower in CONJUNCTIONS:
            tagged.append((token, "CONJ"))
        elif lower in PRONOUNS:
            tagged.append((token, "PRON"))
        elif lower in AUXILIARY_VERBS:
            tagged.append((token, "AUX"))
        elif lower in COMMON_VERBS:
            tagged.append((token, "VERB"))
        elif lower in COMMON_ADJECTIVES:
            tagged.append((token, "ADJ"))
        elif lower in COMMON_ADVERBS:
            tagged.append((token, "ADV"))

        # Verb patterns (morphological)
        elif re.match(r'.*(?:ed|ing|ize|ise|ify|ate)$', lower) and lower not in COMMON_ADJECTIVES:
            tagged.append((token, "VERB"))

        # Adjective patterns
        elif re.match(r'.*(?:ful|less|ous|ive|able|ible|ical|ish)$', lower):
            tagged.append((token, "ADJ"))

        # Adverb pattern
        elif lower.endswith("ly") and len(lower) > 3:
            tagged.append((token, "ADV"))

        # Proper noun (capitalized, not first word or after period)
        elif token[0].isupper() and len(token) > 1:
            tagged.append((token, "PROPN"))

        # Default to noun
        else:
            tagged.append((token, "NOUN"))

    return tagged


def extract_noun_phrases(tagged_tokens: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
    """Extract simple noun phrases from tagged tokens."""
    phrases = []
    current_phrase = []

    for token, tag in tagged_tokens:
        if tag in ("PROPN", "NOUN", "ADJ", "DET", "NUM", "DATE"):
            current_phrase.append((token, tag))
        else:
            if current_phrase:
                # Only keep phrases that have at least one noun/proper noun
                if any(t[1] in ("PROPN", "NOUN") for t in current_phrase):
                    phrases.append(current_phrase)
                current_phrase = []

    if current_phrase and any(t[1] in ("PROPN", "NOUN") for t in current_phrase):
        phrases.append(current_phrase)

    return phrases
