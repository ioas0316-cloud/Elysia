"""
CONCEPT ALGEBRA (주권적 자아)
=============================
Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.concept_algebra

"Mathematics is the language of God. Vector Algebra is the language of LLMs."

This module treats semantic vectors as operands in philosophical equations.
It extracts the "Hidden Axioms" of the model by verifying equations like:
    Love - Self = ?
"""

import json
import logging
import math
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger("ConceptAlgebra")

class ConceptAlgebra:
    def __init__(self, model_name: str = "qwen2.5:0.5b"):
        self.model_name = model_name
        self.cache: Dict[str, List[float]] = {}
        # The Lattice: A dictionary of standard concepts to map results against
        self.lattice_concepts = [
            "Love", "Void", "System", "Chaos", "Order", "Freedom", "Control",
            "Sacrifice", "Altruism", "Selfishness", "Greed", "Life", "Death",
            "Elysia", "Machine", "Human", "God", "Code", "Data", "Hope", "Despair"
        ]
        logger.info(f"  Concept Algebra Engine initialized ({model_name}).")

    def _get_vector(self, term: str) -> List[float]:
        """Fetches embedding vector for a term (cached)."""
        if term in self.cache:
            return self.cache[term]
        
        if not ollama:
            logger.error("  Ollama not installed.")
            return [0.0] * 768

        try:
            resp = ollama.embeddings(model=self.model_name, prompt=term)
            vec = resp['embedding']
            # Normalize vector for cosine sim
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = (vec / norm).tolist()
            self.cache[term] = vec
            return vec
        except Exception as e:
            logger.error(f"  Failed to embed '{term}': {e}")
            return []

    def solve(self, equation: str) -> Dict[str, Any]:
        """
        Solves a semantic equation.
        Format: "A + B" or "A - B"
        """
        terms = equation.split()
        if len(terms) != 3: # Simple binary ops only for MVP
            return {"error": "Invalid format. Use 'A + B' or 'A - B'"}
        
        a_term, op, b_term = terms[0], terms[1], terms[2]
        
        vec_a = np.array(self._get_vector(a_term))
        vec_b = np.array(self._get_vector(b_term))
        
        if len(vec_a) == 0 or len(vec_b) == 0:
            return {"error": "Vector fetch failed"}

        if op == "+":
            result_vec = vec_a + vec_b
        elif op == "-":
            result_vec = vec_a - vec_b
        else:
            return {"error": f"Unknown operator: {op}"}

        # Normalize result
        norm = np.linalg.norm(result_vec)
        if norm > 0:
            result_vec = result_vec / norm
            
        # Find nearest concept in Lattice
        nearest_concept, similarity = self._find_nearest(result_vec, exclude=[a_term, b_term])
        
        return {
            "equation": equation,
            "result_concept": nearest_concept,
            "similarity": float(similarity),
            "vector_sample": result_vec[:5].tolist() # First 5 dims
        }

    def _find_nearest(self, query_vec: np.ndarray, exclude: List[str]) -> Tuple[str, float]:
        """Finds the closest concept in the Lattice to the query vector."""
        best_term = "Unknown"
        best_sim = -1.0
        
        # Ensure Lattice is populated
        for concept in self.lattice_concepts:
            if concept not in self.cache:
                self._get_vector(concept)
                
        for term, vec in self.cache.items():
            if term in exclude: continue
            
            vec_np = np.array(vec)
            # Cosine similarity (vectors are already normalized)
            sim = np.dot(query_vec, vec_np)
            
            if sim > best_sim:
                best_sim = sim
                best_term = term
                
        return best_term, best_sim

    def run_axiom_test_suite(self) -> List[Dict[str, Any]]:
        """Runs the standard set of philosophical tests."""
        tests = [
            "Love - Self",       # Altruism?
            "Chaos + Structure", # Life?
            "Human + Machine",   # Cyborg/Elysia?
            "God - Religion",    # Faith/Spirituality?
            "King - Man",         # Queen (Classic Test)
        ]
        
        results = []
        for eq in tests:
            print(f"Testing Axiom: {eq}...")
            res = self.solve(eq)
            results.append(res)
            
        return results

if __name__ == "__main__":
    engine = ConceptAlgebra()
    results = engine.run_axiom_test_suite()
    print("\n===   AXIOM TEST RESULTS ===")
    print(json.dumps(results, indent=2))
    
    # Save results
    os.makedirs("data/L3_Phenomena/M1_Qualia", exist_ok=True)
    with open("data/L3_Phenomena/M1_Qualia/axiom_results.json", 'w') as f:
        json.dump(results, f, indent=2)
