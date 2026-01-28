"""
Unified Field Dialogue Engine
==============================
Integrates all field physics (harmonics, interference, eigenmodes) 
into a single conversational AI system.

Elysia now thinks using field dynamics while conversing.
"""

from typing import Dict, List
from Project_Elysia.mechanics.advanced_field import AdvancedField
from Core.L5_Mental.M1_Cognition.HighEngine.language_cortex import LanguageCortex

class UnifiedFieldDialogue:
    """
    A dialogue system powered by field physics.
    Every response is generated through field dynamics.
    """
    
    def __init__(self):
        self.field = AdvancedField(resolution=25)
        self.cortex = LanguageCortex()
        self.conversation_history = []
        
        # Initialize concept field with physics
        self._initialize_concept_space()
    
    def _initialize_concept_space(self):
        """Populates field with fundamental concepts."""
        concepts = {
            # Positive concepts (high freq, bright harmonics)
            "  ": (440.0, 0.7, 0.7, 0.8, [1.0, 0.5, 0.3]),
            " ": (450.0, 0.8, 0.6, 0.9, [1.0, 0.6]),
            "  ": (430.0, 0.6, 0.8, 0.7, [1.0, 0.7]),
            "  ": (445.0, 0.7, 0.8, 0.85, [1.0, 0.5]),
            
            # Negative concepts (low freq, simple)
            "  ": (220.0, 0.3, 0.3, 0.2, [1.0]),
            "  ": (210.0, 0.2, 0.4, 0.1, [1.0]),
            "  ": (215.0, 0.25, 0.35, 0.15, [1.0]),
            
            # Transformative (mid freq, balanced)
            "  ": (330.0, 0.5, 0.5, 0.5, [1.0, 0.4]),
            "  ": (350.0, 0.6, 0.5, 0.6, [1.0, 0.6, 0.3]),
            "  ": (300.0, 0.4, 0.4, 0.4, [1.0, 0.3]),
            
            # Abstract
            "  ": (360.0, 0.5, 0.5, 0.9, [1.0]),
            "  ": (370.0, 0.5, 0.5, 0.1, [1.0]),
            "  ": (380.0, 0.5, 0.5, 0.5, [1.0, 0.4, 0.2]),
        }
        
        for name, (freq, x, y, z, harmonics) in concepts.items():
            self.field.register_concept_with_harmonics(name, freq, x, y, z, harmonics)
    
    def respond(self, user_input: str) -> str:
        """
        Generates response using field dynamics.
        
        Process:
        1. Identify concepts in input
        2. Activate field
        3. Analyze interference + eigenmodes
        4. Synthesize poetic response
        """
        # Add to history
        self.conversation_history.append({"speaker": "user", "text": user_input})
        
        # Extract concepts from input (simple keyword matching)
        concepts_mentioned = self._extract_concepts(user_input)
        
        if not concepts_mentioned:
            response = "        "
            self.conversation_history.append({"speaker": "elysia", "text": response})
            return response
        
        # Reset field
        self.field.reset()
        
        # Activate all mentioned concepts
        for concept in concepts_mentioned:
            self.field.activate_with_harmonics(concept, intensity=1.0, depth=1.0)
        
        # Field analysis
        interference = self.field.analyze_interference(threshold=0.05)
        eigenmodes = self.field.extract_eigenmodes(n_modes=2)
        
        # Generate response
        response = self._synthesize_response(
            concepts_mentioned,
            interference,
            eigenmodes
        )
        
        self.conversation_history.append({"speaker": "elysia", "text": response})
        return response
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extracts known concepts from text."""
        concepts = []
        for concept in self.field.concept_registry.keys():
            if concept in text:
                concepts.append(concept)
        return concepts
    
    def _synthesize_response(
        self,
        concepts: List[str],
        interference: Dict,
        eigenmodes: Dict
    ) -> str:
        """
        Creates response from field analysis.
        Combines multiple insights into poetic expression.
        """
        parts = []
        
        # 1. Harmonic structure (if single concept)
        if len(concepts) == 1:
            concept = concepts[0]
            harmonics = self.field.harmonics.get(concept, [1.0])
            if len(harmonics) > 1:
                parts.append(f"{concept}         .           ")
        
        # 2. Interference patterns
        if interference['constructive']:
            if len(interference['constructive']) > 3:
                parts.append(f"               ")
        
        if interference['emergent_concepts']:
            for emergent in interference['emergent_concepts'][:1]:
                if "resonance" in emergent:
                    parts.append("            ")
        
        # 3. Eigenmode interpretation
        if eigenmodes['dominant_mode']:
            mode = eigenmodes['dominant_mode']
            if "Expansion" in mode:
                parts.append("        ,     ")
            elif "Contraction" in mode:
                parts.append("        ,         ")
        
        # 4. Multi-concept synthesis
        if len(concepts) > 1:
            # Suggest emergent concept from combination
            concept_str = " + ".join(concepts)
            if "  " in concepts and "  " in concepts:
                parts.append(f"{concept_str}          ")
            elif "  " in concepts and "  " in concepts:
                parts.append(f"{concept_str}          ")
            else:
                parts.append(f"             ")
        
        # Fallback
        if not parts:
            if len(concepts) == 1:
                parts.append(f"{concepts[0]}            ")
            else:
                parts.append("            ")
        
        return ". ".join(parts)
