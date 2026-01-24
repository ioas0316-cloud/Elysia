import numpy as np
from typing import Dict, Optional, Any
from Core.L1_Foundation.Logic.d7_vector import D7Vector
from Core.L1_Foundation.Logic.qualia_7d_codec import codec

class QualiaProjector:
    """
    [STEEL CORE] Qualia Projector
    =============================
    Projects fuzzy instructions/data into the strict D7 Manifold.
    """
    
    def __init__(self):
        self.keywords = {
            "Foundation": ["structure", "basis", "save", "load", "file", "root", "database", "disk"],
            "Metabolism": ["fast", "pulse", "cycle", "loop", "heartbeat", "speed", "frequency", "flow"],
            "Phenomena": ["see", "display", "print", "show", "interface", "ui", "log", "sense"],
            "Causality": ["predict", "fate", "roadmap", "plan", "future", "history", "because", "link"],
            "Mental": ["logic", "reason", "think", "analyze", "math", "code", "if", "else", "algorithm"],
            "Structure": ["engine", "architecture", "merkaba", "module", "class", "function", "pattern"],
            "Spirit": ["love", "merciful", "genesis", "sovereign", "will", "intent", "purpose", "grace"]
        }

    def project_instruction(self, text: str) -> D7Vector:
        """
        Projects a natural language instruction into D7 space.
        Returns a D7Vector with normalized intensities.
        """
        intensities = {layer: 0.05 for layer in self.keywords.keys()} # Baseline noise
        
        text_lower = text.lower()
        for layer, words in self.keywords.items():
            for word in words:
                if word in text_lower:
                    intensities[layer] += 0.4
                    
        # Cap and return
        return D7Vector(
            foundation=min(1.0, intensities["Foundation"]),
            metabolism=min(1.0, intensities["Metabolism"]),
            phenomena=min(1.0, intensities["Phenomena"]),
            causality=min(1.0, intensities["Causality"]),
            mental=min(1.0, intensities["Mental"]),
            structure=min(1.0, intensities["Structure"]),
            spirit=min(1.0, intensities["Spirit"])
        )

# Global Instance
projector = QualiaProjector()
