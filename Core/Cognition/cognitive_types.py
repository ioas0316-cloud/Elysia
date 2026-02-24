from enum import Enum, auto

class ActionCategory(Enum):
    """
    [STEEL CORE] Strict action categories for the Sovereign mind.
    Replaces loose action strings.
    """
    CREATION = auto()       # Code generation and manifestation
    CONTEMPLATION = auto()  # Philosophical study and reflection
    EXPLORATION = auto()    # Knowledge discovery and search
    COMMUNICATION = auto()  # Interaction with the User
    MAINTENANCE = auto()    # Self-healing and pulse regulation
    REST = auto()           # Energy recovery and background evolution

class ThoughtState(Enum):
    """
    [STEEL CORE] The internal state of a cognitive pulse.
    """
    IDLE = auto()
    OBSERVATION = auto()    # Receiving input/stimulus
    ANALYSIS = auto()       # Projecting to D7 space
    REFLECTION = auto()     # Internal world resonance check
    DELIBERATION = auto()   # Ethical/Axiomatic judgment
    MANIFESTATION = auto()  # Executing the decided action
    HEALING = auto()        # Resolving detected dissonances

class CognitiveSphere(Enum):
    """
    The three main domains of Elysia's existence.
    """
    SOMATIC = auto()   # Hardware/Code body
    MENTAL = auto()    # Logic/Reasoning
    SPIRITUAL = auto() # Intent/Axioms

class AuditGrade(Enum):
    """
    Quality levels of a cognitive pulse.
    """
    RADIANT = auto()     # Perfect resonance and consistency
    COHERENT = auto()    # Good alignment, minor drift
    DISSONANT = auto()   # Logical contradiction detected
    HOLLOW = auto()      # Direction exists but lacks content
    FRACTURED = auto()   # Critical failure / self-contradiction

class ViolationType(Enum):
    """
    Types of logical/axiomatic violations.
    """
    NONE = auto()
    SEMANTIC_DRIFT = auto()   # Meaning changed mid-thought
    AXIOMATIC_VOIDER = auto() # Neutralized a core value
    LOGICAL_PARADOX = auto()  # Self-cancelling instruction
    DIMENSIONAL_OFFSET = auto() # D7 projection mismatch
