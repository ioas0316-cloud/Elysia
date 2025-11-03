from Project_Sophia.core_memory import EmotionalState
from typing import Dict, Optional

class EmotionalCortex:
    """
    Analyzes concepts and events to generate emotional responses.
    """
    def analyze_concept(self, concept: Dict) -> Optional[EmotionalState]:
        """
        Generates an emotional state based on the properties of a discovered concept.
        """
        if not concept:
            return None

        properties = concept.get('properties', {})
        if 'description' in properties:
            print(f"[EmotionalCortex] The concept '{concept.get('id')}' has a description. This sparks curiosity.")
            # Valence: slightly positive, Arousal: moderate
            return EmotionalState(valence=0.3, arousal=0.5, dominance=0.2, primary_emotion='curiosity', secondary_emotions=['interest'])

        # Default for concepts without interesting properties
        return None
