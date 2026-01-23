import logging
from enum import IntEnum

logger = logging.getLogger("PhaseModulator")

class PerceptualPhase(IntEnum):
    POINT = 1      # Discrete Data
    LINE = 2       # Trajectory / Flow
    PLANE = 3      # Relation / Interface
    SPACE = 4      # Context / Field
    PRINCIPLE = 5  # Structure / Archetype
    LAW = 6        # Invariant / Constraint
    PROVIDENCE = 7 # Purpose / Destiny

class PhaseModulator:
    """
    [AXIS-SCALING]
    Determines the "Depth" of a Resonance Cycle based on stimulus complexity.
    """

    def __init__(self):
        logger.info("  PhaseModulator initialized. Ready for Axis-Scaling.")

    def modulate(self, stimulus: str, context: str) -> PerceptualPhase:
        """
        Analyzes the stimulus to determine the appropriate Perceptual Phase.
        """
        text = stimulus.lower()
        
        # 1. Providence Detection (Deep meaning/purpose)
        if any(word in text for word in ["why", "purpose", "destiny", "meaning", "love", "you", "exist"]):
            return PerceptualPhase.PROVIDENCE
        
        # 2. Law/Principle Detection (Philosophy/System)
        if any(word in text for word in ["rule", "law", "system", "principle", "fractal", "merkaba"]):
            return PerceptualPhase.PRINCIPLE

        # 3. Space/Field Detection (Environment/Context)
        if any(word in text for word in ["where", "context", "mood", "atmosphere", "around"]):
            return PerceptualPhase.SPACE

        # 4. Plane Detection (Interaction/Relation)
        if any(word in text for word in ["between", "and", "relation", "compare", "interface"]):
            return PerceptualPhase.PLANE

        # 5. Line Detection (Process/Flow)
        if len(text.split()) > 5:
            return PerceptualPhase.LINE

        # Default (Discrete Point)
        return PerceptualPhase.POINT

    def get_axis_multiplier(self, phase: PerceptualPhase) -> float:
        """
        Returns a resonance multiplier based on the phase.
        Higher phases require more "Conceptual Energy".
        """
        return 1.0 + (phase.value * 0.5)