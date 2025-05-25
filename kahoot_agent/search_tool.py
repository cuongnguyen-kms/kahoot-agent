import json
import os
from typing import List, Dict, Any
from langchain_core.tools import tool

try:
    from sentence_transformers import SentenceTransformer, util
    _has_st = True
except ImportError:
    _has_st = False

# Path to the knowledge base JSON file (should be placed in the project root or kahoot_agent/)
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), 'kahoot_knowledge.json')

# Load knowledge base
if os.path.exists(KNOWLEDGE_PATH):
    with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
        KNOWLEDGE = json.load(f)
else:
    KNOWLEDGE = []

# Prepare texts for embedding
KB_TEXTS = [item['text'] if 'text' in item else str(item) for item in KNOWLEDGE]

# Load embedding model if available
if _has_st:
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    KB_EMBEDDINGS = MODEL.encode(KB_TEXTS, convert_to_tensor=True)
else:
    MODEL = None
    KB_EMBEDDINGS = None

@tool
def search_knowledge(query: str, method: str = 'keyword', top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search the Kahoot knowledge base using either 'keyword' or 'vector' method.
    Args:
        query: The search query string.
        method: 'keyword' for substring match, 'vector' for semantic search (default: 'keyword').
        top_k: Number of top results to return (default: 3).
    Returns:
        List of top matching knowledge entries.
    """
    if not KNOWLEDGE:
        return []
    if method == 'vector' and _has_st and MODEL is not None:
        query_emb = MODEL.encode(query, convert_to_tensor=True)
        hits = util.cos_sim(query_emb, KB_EMBEDDINGS)[0]
        top_indices = hits.argsort(descending=True)[:top_k]
        return [KNOWLEDGE[i] for i in top_indices]
    else:
        # Simple keyword search (case-insensitive substring)
        matches = [item for item in KNOWLEDGE if query.lower() in (item.get('text', str(item)).lower())]
        # Fallback: if not enough matches, return closest by fuzzy match
        if len(matches) < top_k:
            try:
                from difflib import get_close_matches
                texts = [item.get('text', str(item)) for item in KNOWLEDGE]
                close = get_close_matches(query, texts, n=top_k)
                matches += [item for item in KNOWLEDGE if item.get('text', str(item)) in close and item not in matches]
            except ImportError:
                pass
        return matches[:top_k]
