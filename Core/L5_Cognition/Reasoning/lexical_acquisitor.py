"""
Lexical Acquisitor (The Scholar)
================================
"To name a thing is to define its resonance."

This module scans memory for unknown terms and attempts to 
infer their 21D principle vectors based on surrounding context.
"""

import jax.numpy as jnp
from typing import List, Dict, Optional, Any
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge
from Core.L5_Cognition.Reasoning.inferential_manifold import InferentialManifold

class LexicalAcquisitor:
    def __init__(self):
        self.manifold = InferentialManifold()
        self.ignored_words = {
            "the", "and", "a", "is", "in", "it", "to", "of", "with", "for", "as", "on", "at", 
            "concept", "phase", "status", "mass", "type", "description", "vector", "logic",
            "cycle", "rotor", "monad", "pulse", "state", "engine", "module", "core", "path",
            "field", "torque", "rpm", "damping", "mass", "mass:", "torque:", "phase:",
            "truth", "love", "void", "spirit", "arcadia", "idyll", "boundary", "edge",
            "motion", "life", "agape", "logos", "soma", "nunchi", "merkaba", "dna",
            "resonance", "alignment", "equilibrium", "consensus", "leader", "nunchi",
            "nunchi:", "governor", "exhalation", "inhalation", "volume", "stagnation",
            "threshold", "reconfigurator", "manifold", "underworld", "identity",
            "system", "think", "know", "will", "does", "that", "this", "they", "your", "have",
            "been", "into", "from", "their", "process", "could", "would", "which", "when"
        }

    def scan_for_learning(self, memories: List[Any]) -> Optional[str]:
        """
        Looks for frequent/hot words in memory that aren't mapped.
        Returns a candidate word to learn.
        """
        word_freq = {}
        for node in memories:
            # We skip 'Archeology' nodes for high-speed scan
            if node.temperature < 30.0: continue
            
            words = node.content.lower().split()
            for w in words:
                # [PHASE 63] Better Cleaning: Remove markdown and structural tags
                if w.startswith("[") or w.endswith("]"): continue
                w = w.strip(".,?!:;\"'()[]{}*-_") # Added markdown symbols
                
                if len(w) < 4: continue
                if w in self.ignored_words: continue
                if any(char.isdigit() for char in w): continue # Skip versions/numbers
                
                # Check if we already know it
                if jnp.any(LogosBridge.recall_concept_vector(w)):
                    continue
                
                word_freq[w] = word_freq.get(w, 0) + (node.mass * 0.1)
                
        if not word_freq:
            print("🔍 [DEBUG] No unknown words found in hot memory.")
            return None
            
        # Pick the most 'resonant' unknown word
        best_word = max(word_freq, key=word_freq.get)
        print(f"🔍 [DEBUG] Candidate: '{best_word}' (Freq: {word_freq[best_word]:.2f})")
        if word_freq[best_word] > 2.0: # Threshold for learning
            return best_word
        return None

    def attempt_acquisition(self, word: str, context_memories: List[Any]):
        """
        Uses surrounding context to guess the word's principle vector.
        """
        accumulated_field = jnp.zeros(21)
        samples = 0
        
        for node in context_memories:
            if word.lower() in node.content.lower():
                # Extract the Intent of the sentence EXCLUDING the word itself
                # (Conceptual proximity)
                intent = LogosBridge.calculate_text_resonance(node.content.replace(word, ""))
                if jnp.any(intent):
                    accumulated_field += intent
                    samples += 1
        
        if samples == 0:
            return False
            
        vector = accumulated_field / (jnp.linalg.norm(accumulated_field) + 1e-6)
        
        # Register the new concept
        LogosBridge.learn_concept(
            name=word,
            vector=vector,
            description=f"Acquired through Epistemic learning cycle. Base Resonance: {samples} samples."
        )
        return True
