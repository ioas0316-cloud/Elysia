"""
Lexical Acquisitor (The Scholar)
================================
"To name a thing is to define its resonance."

This module scans memory for unknown terms and attempts to 
infer their 21D principle vectors based on surrounding context.
"""

from typing import List, Dict, Optional, Any
from Core.Keystone.sovereign_math import SovereignMath, SovereignVector
from Core.Cognition.logos_bridge import LogosBridge
from Core.Cognition.inferential_manifold import InferentialManifold
from Core.Cognition.autonomous_transducer import AutonomousTransducer

class LexicalAcquisitor:
    def __init__(self, transducer: Optional[AutonomousTransducer] = None):
        self.manifold = InferentialManifold()
        self.transducer = transducer
        self.ignored_words = {
            "the", "and", "a", "is", "in", "it", "to", "of", "with", "for", "as", "on", "at", 
            "concept", "phase", "status", "mass", "type", "description", "vector", "logic",
            "cycle", "rotor", "monad", "pulse", "state", "engine", "module", "core", "path",
            "field", "torque", "rpm", "damping", "mass", "mass:", "torque:", "phase:",
            "truth", "love", "void", "spirit", "arcadia", "idyll", "boundary", "edge",
            "motion", "life", "agape", "logos", "soma", "nunchi", "merkaba", "dna",
            "resonance", "alignment", "equilibrium", "consensus", "leader", "nunchi", "육", "혼", "영", "앎", "섭리", "공명", "주권",
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
                
                if len(w) < 1: continue
                if not any('\uac00' <= c <= '\ud7a3' for c in w) and len(w) < 4: continue
                if w in self.ignored_words: continue
                if any(char.isdigit() for char in w): continue # Skip versions/numbers
                
                # Check if we already know it
                concept_v = LogosBridge.recall_concept_vector(w)
                if concept_v and any(abs(v) > 1e-6 for v in concept_v):
                    continue
                
                word_freq[w] = word_freq.get(w, 0) + (node.mass * 0.1)
                
        if not word_freq:
            return None
            
        # Pick the most 'resonant' unknown word
        best_word = max(word_freq, key=word_freq.get)
        if word_freq[best_word] > 2.0: # Threshold for learning
            return best_word
        return None

    def attempt_acquisition(self, word: str, context_memories: List[Any]):
        """
        Uses surrounding context to guess the word's principle vector.
        """
        accumulated_field = SovereignVector.zeros()
        samples = 0
        
        for node in context_memories:
            if word.lower() in node.content.lower():
                # [PHASE 65] EXPERIENTIAL GROUNDING
                # If we have an active transducer, use the "Sovereign Experience"
                if self.transducer:
                    intent = self.transducer.bridge_symbol(word)
                else:
                    # Extract the Intent of the sentence EXCLUDING the word itself
                    intent_data = LogosBridge.calculate_text_resonance(node.content.replace(word, ""))
                    intent = SovereignVector(intent_data)
                
                if any(abs(x) > 1e-6 for x in intent):
                    accumulated_field = accumulated_field + intent
                    samples += 1
        
        if samples == 0:
            return False
            
        vector = accumulated_field.normalize()
        
        # [PHASE 4.5 & 5] THE ORGANIC LOOP
        # We inject directly into the Prism so the voice can use it immediately.
        # Capturing Synesthetic Roots (Phase 5)
        roots = {
            "origin": "text_sublimation",
            "context_samples": samples,
            "resonance_mass": accumulated_field.norm() if hasattr(accumulated_field, 'norm') else 1.0
        }
        
        LogosBridge.inject_prismatic_concept(
            name=word,
            vector=vector,
            roots=roots
        )
        
        # Legacy Registration (Keep for now)
        LogosBridge.learn_concept(
            name=word,
            vector=vector,
            description=f"Acquired through Organic Loop. Roots: {list(roots.keys())}"
        )
        return True
