"""
Qualia Transducer: The Sense Maker
==================================
Core.L4_Causality.World.Physics.qualia_transducer

"To see the world is to measure its resonance against the Absolute."

This module converts raw information (Text/Structure) into 
7D Trinity Vectors (Qualia) using a true Semantic Embedding model.

Mechanism:
1.  **Sensation**: Embeds text into 384D vector using `all-MiniLM-L6-v2`.
2.  **Projection**: Projects 384D -> 7D using 'Anchor Resonance'.
    - We have 7 Semantic Anchors (Wisdom, Hope, Faith, etc.) with known 7D coords.
    - The new concept's 7D coord is the weighted average of Anchors,
      weighted by their cosine similarity in 384D space.

This ensures that "Love" (Text) -> [0, 0, 0] (Vector) is functionally true 
because it resonates max with the "Love" anchor.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from Core.L4_Causality.World.Physics.trinity_fields import TrinityVector

# Soft import
try:
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("QualiaTransducer")

class QualiaTransducer:
    def __init__(self):
        self.device = "cpu" # MiniLM is small enough for CPU
        self.model = None
        self.anchors_384d = {} # {name: tensor(384)}
        
        # known 7D coordinates for Anchors
        # (Harmony, Energy, Inspiration) - mapped to TrinityVector
        self.anchors_7d = {
            "Love":       TrinityVector(0.0, 0.0, 0.0),    # The One / Void
            "Wisdom":     TrinityVector(0.9, 0.5, 0.9),    # High Inspiration & Harmony
            "Hope":       TrinityVector(0.5, 0.8, 0.9),    # High Energy & Inspiration
            "Faith":      TrinityVector(0.8, 0.2, 1.0),    # Max Inspiration, High Harmony
            "Courage":    TrinityVector(0.2, 0.9, 0.4),    # Max Energy
            "Justice":    TrinityVector(0.9, 0.6, 0.1),    # High Harmony, Low Inspiration (Logic)
            "Temperance": TrinityVector(0.7, 0.1, 0.3),    # Low Energy, High Harmony
            "Truth":      TrinityVector(0.0, 0.0, 1.0),    # Pure Spirit
            
            # Demons (Negative Space)
            "Chaos":      TrinityVector(-0.9, 0.9, -0.9),  # High Energy, Negative Harmony
            "Void":       TrinityVector(0.0, -0.5, 0.0),   # Low Energy
            "Fear":       TrinityVector(-0.5, -0.2, -0.5)  # Negative
        }
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("📡 Loading Sensory Cortex (all-MiniLM-L6-v2)...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self._embed_anchors()
                logger.info("✅ Sensory Cortex Calibrated.")
            except Exception as e:
                logger.error(f"Failed to load Sensory Cortex: {e}")
        else:
            logger.warning("⚠️ sentenced-transformers not installed. Using simple hash.")

    def _embed_anchors(self):
        """Pre-calculates 384D vectors for all anchors."""
        names = list(self.anchors_7d.keys())
        embeddings = self.model.encode(names, convert_to_tensor=True)
        
        for name, emb in zip(names, embeddings):
            self.anchors_384d[name] = emb

    def transduce(self, text: str) -> TrinityVector:
        """
        Converts text -> 7D TrinityVector.
        """
        if not self.model:
            return self._heuristic_transduce(text)
            
        # 1. Embed Input
        input_emb = self.model.encode(text, convert_to_tensor=True)
        
        # 2. Calculate Similarity with Anchors
        anchor_names = list(self.anchors_384d.keys())
        anchor_embs = torch.stack([self.anchors_384d[n] for n in anchor_names])
        
        # Cosine Similarity
        # (1, 384) x (N, 384).T -> (1, N)
        sim_scores = util.cos_sim(input_emb, anchor_embs)[0]
        
        # 3. Weighted Average of 7D Anchors
        # We emphasize the top matches (Linear Projection is too noisy)
        # Softmax-like weighting? Or just verify top 3.
        
        # Let's take Top 3 anchors
        top_k = 3
        top_indices = torch.topk(sim_scores, k=top_k).indices
        
        final_vec = TrinityVector(0,0,0)
        total_weight = 0.0
        
        for idx in top_indices:
            score = sim_scores[idx].item()
            if score < 0.1: continue # Ignore non-resonance
            
            name = anchor_names[idx]
            anchor_vec = self.anchors_7d[name]
            
            # Weighting
            weight = score ** 2 # Emphasize strong matches
            
            final_vec = final_vec + (anchor_vec * weight)
            total_weight += weight
            
        if total_weight > 0:
            final_vec = final_vec * (1.0 / total_weight)
            
        return final_vec

    def _heuristic_transduce(self, text: str) -> TrinityVector:
        """Fallback if model unavailable."""
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Normalize to -1.0 to 1.0
        x = ((h % 1000) / 500.0) - 1.0
        y = (((h // 1000) % 1000) / 500.0) - 1.0
        z = (((h // 1000000) % 1000) / 500.0) - 1.0
        return TrinityVector(x, y, z)

# Singleton
_transducer = None
def get_qualia_transducer():
    global _transducer
    if _transducer is None:
        _transducer = QualiaTransducer()
    return _transducer
