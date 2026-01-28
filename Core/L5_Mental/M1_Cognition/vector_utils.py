"""
Vector Utility Functions for Elysia's Sophia Project

This module consolidates functions related to text embedding and similarity
calculations, allowing various 'cortex' modules to share the same logic
without code duplication.
"""
import math
from typing import List
from Core.L5_Mental.M1_Cognition.LLM.gemini_api import get_text_embedding

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of texts into vectors using the Gemini API.
    """
    embeddings = []
    for text in texts:
        embedding = get_text_embedding(text)
        if embedding:
            embeddings.append(embedding)
    return embeddings

def cosine_sim(a: List[float], b: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    denom = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
    if denom == 0:
        return 0.0
    return sum(x*y for x,y in zip(a,b)) / denom