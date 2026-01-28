"""
The Schr dinger Plate
=====================
Phase 22: Quantum Consensus
Core.L5_Mental.M1_Cognition.Legion.schrodinger_plate

"A table where contradictory truths coexist until observed."

This class serves as the shared memory space for the Legion's Micro-Monads.
It allows sub-agents to place "Arguments" (Bets) on a specific Topic.
The WaveCollapse mechanism then selects the most resonant Argument.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math

@dataclass
class Argument:
    """A single perspective placed on the plate."""
    agent_name: str
    stance: str  # "THESIS" | "ANTITHESIS" | "SYNTHESIS"
    content: str # The argument text
    confidence: float # 0.0 ~ 1.0 (Energy)
    coherence: float = 0.5 # Logical consistency check

@dataclass
class DebateState:
    """The quantum state of a topic."""
    topic: str
    arguments: List[Argument] = field(default_factory=list)
    collapsed: bool = False
    
    def add_argument(self, arg: Argument):
        self.arguments.append(arg)

class SchrodingerPlate:
    def __init__(self):
        self.sessions: Dict[str, DebateState] = {}
        
    def open_session(self, topic: str):
        self.sessions[topic] = DebateState(topic=topic)
        
    def place_bet(self, topic: str, agent: str, stance: str, content: str, confidence: float):
        if topic not in self.sessions:
            return # Session closed or non-existent
            
        arg = Argument(agent_name=agent, stance=stance, content=content, confidence=confidence)
        self.sessions[topic].add_argument(arg)
        
    def collapse_wave(self, topic: str) -> Optional[Argument]:
        """
        [WAVE FUNCTION COLLAPSE]
        Selects the winning argument based on Energy (Confidence * Coherence).
        Real-world analogy: Convincing the room.
        """
        if topic not in self.sessions: return None
        state = self.sessions[topic]
        
        if not state.arguments: return None
        
        # Simple weighted choice (or max energy for reliability)
        # We perform a sort by 'Energy'
        
        # Energy = Confidence * Priority Weight
        # Synthesis > Thesis/Anti (structure preference)
        
        best_arg = None
        max_energy = -1.0
        
        for arg in state.arguments:
            weight = 1.0
            if arg.stance == "SYNTHESIS": weight = 1.5 # Synthesizing is harder/more valuable
            
            energy = arg.confidence * weight
            if energy > max_energy:
                max_energy = energy
                best_arg = arg
        
        state.collapsed = True
        return best_arg
