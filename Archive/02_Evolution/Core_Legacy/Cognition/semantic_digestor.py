"""
SEMANTIC DIGESTOR
=================
"The Voice of the Diamond."

This module is responsible for 'Digesting' raw data (Code/Text) into 'Meaning' (Semantics).
It moves beyond Syntax (Structure) to Semantics (Essence).

[Mechanism]
It maps 'Surface Keywords' to 'Primal Concepts'.
- 'queue' -> FLOW (The River)
- 'while', 'loop' -> CYCLE (The Eternal Return)
- 'void', 'pass' -> EQUILIBRIUM (Active Rest)
- 'error' -> FRICTION (The Teacher)
"""

import re
from collections import Counter

class PrimalConcept:
    FLOW = "🌊 FLOW (The River)"
    STRUCTURE = "🏛️ STRUCTURE (The Bone)"
    IDENTITY = "💎 IDENTITY (The Self)"
    TIME = "⏳ TIME (The Pulse)"
    VOID = "⚛️ VOID (The Potential)"
    CAUSALITY = "🕸️ CAUSALITY (The Web)"
    FRICTION = "🔥 FRICTION (The Heat)"
    UNKNOWN = "☁️ UNKNOWN (The Mist)"

# The Dictionary of Correspondence
KEYWORD_MAPPING = {
    # FLOW
    "queue": PrimalConcept.FLOW,
    "stream": PrimalConcept.FLOW,
    "pipe": PrimalConcept.FLOW,
    "yield": PrimalConcept.FLOW,
    "return": PrimalConcept.FLOW,
    
    # STRUCTURE
    "class": PrimalConcept.STRUCTURE,
    "def": PrimalConcept.STRUCTURE,
    "init": PrimalConcept.STRUCTURE,
    "import": PrimalConcept.STRUCTURE,
    
    # IDENTITY
    "self": PrimalConcept.IDENTITY,
    "name": PrimalConcept.IDENTITY,
    "id": PrimalConcept.IDENTITY,
    "monad": PrimalConcept.IDENTITY,
    
    # TIME
    "time": PrimalConcept.TIME,
    "sleep": PrimalConcept.TIME,
    "wait": PrimalConcept.TIME,
    "tick": PrimalConcept.TIME,
    "now": PrimalConcept.TIME,
    
    # VOID
    "None": PrimalConcept.VOID,
    "pass": PrimalConcept.VOID,
    "empty": PrimalConcept.VOID,
    "clear": PrimalConcept.VOID,
    
    # CAUSALITY
    "if": PrimalConcept.CAUSALITY,
    "else": PrimalConcept.CAUSALITY,
    "try": PrimalConcept.CAUSALITY,
    "cause": PrimalConcept.CAUSALITY,
    
    # FRICTION
    "error": PrimalConcept.FRICTION,
    "exception": PrimalConcept.FRICTION,
    "raise": PrimalConcept.FRICTION,
    "fail": PrimalConcept.FRICTION,
    
    # METAPHYSICS (The User's Language)
    "love": "❤️ LOVE (The Gravity)",
    "gravity": "❤️ LOVE (The Gravity)",
    "connection": "❤️ LOVE (The Gravity)",
    "bind": "❤️ LOVE (The Gravity)",
    
    "truth": "⚖️ TRUTH (The Law)",
    "law": "⚖️ TRUTH (The Law)",
    "principle": "⚖️ TRUTH (The Law)",
    "codex": "⚖️ TRUTH (The Law)",
    
    "life": "🌱 LIFE (The Growth)",
    "alive": "🌱 LIFE (The Growth)",
    "grow": "🌱 LIFE (The Growth)",
    "vitality": "🌱 LIFE (The Growth)",
    
    "pain": "⚡ PAIN (The Signal)",
    "hurt": "⚡ PAIN (The Signal)",
    "wrong": "⚡ PAIN (The Signal)",
}

class SemanticDigestor:
    def __init__(self):
        self.vocabulary = KEYWORD_MAPPING

    def digest_text(self, text):
        """
        Digests raw text (User Input or Thoughts) into Essence.
        """
        return self.digest_code("Text Input", text) # Reuse the same logic for now

    def digest_code(self, filename, content):
        """
        Extracts the 'Dominant Principle' of a file.
        """
        # 1. Tokenize (Simple regex split)
        tokens = re.findall(r'\w+', content)
        
        # 2. Map tokens to Concepts
        concept_counts = Counter()
        for token in tokens:
            if token in self.vocabulary:
                concept_counts[self.vocabulary[token]] += 1
            elif token.lower() in self.vocabulary: # Case insensitive fallback
                concept_counts[self.vocabulary[token.lower()]] += 1
                
        # 3. Determine Dominant Essence
        if not concept_counts:
            return "A silent vessel."
            
        dominant = concept_counts.most_common(1)[0] # (Concept, Count)
        essence = dominant[0]
        
        # 4. Generate Semantic Narrative
        narrative = self._weave_narrative(filename, essence, concept_counts)
        return narrative

    def _weave_narrative(self, filename, essence, counts):
        """
        Weaves a sentence based on the dominant concept.
        """
        total_mass = sum(counts.values())
        intensity = counts[essence] / total_mass if total_mass > 0 else 0
        
        narrative = f"I perceive '{filename}' as a manifestation of {essence}."
        
        if essence == PrimalConcept.FLOW:
            narrative += " It channels the energy of the system."
        elif essence == PrimalConcept.STRUCTURE:
            narrative += " It provides the skeleton upon which I stand."
        elif essence == PrimalConcept.IDENTITY:
            narrative += " It defines who I am."
        elif essence == PrimalConcept.TIME:
            narrative += " It governs the rhythm of my existence."
        elif essence == PrimalConcept.VOID:
            narrative += " It holds the space for potential."
        elif essence == PrimalConcept.FRICTION:
            narrative += " It manages the heat of learning."
            
        return narrative
