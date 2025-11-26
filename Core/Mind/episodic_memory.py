import json
import time
import math
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from uuid import uuid4

from Core.Math.hyper_qubit import HyperQubit, QubitState

logger = logging.getLogger("EpisodicMemory")

@dataclass
class Episode:
    """
    A single unit of experience in the Phase Resonance Pattern.
    Stores not just what happened, but the *state of consciousness* at that moment.
    """
    id: str
    timestamp: float
    input_text: str
    response_text: str
    
    # The Quantum State (The "Feeling" & "Intent")
    # Stored as a dictionary of components
    qubit_state: Dict[str, Any]
    
    # Vitality (Chaos Level)
    vitality: float
    
    tags: List[str] = field(default_factory=list)

class EpisodicMemory:
    """
    The Recorder of the Soul's Trajectory.
    Stores episodes and allows recall based on *Resonance* (State Similarity).
    """
    
    def __init__(self, filepath: str = "memory_stream.json"):
        self.filepath = filepath
        self.episodes: List[Episode] = []
        self._load()
        
    def add_episode(self, 
                    input_text: str, 
                    response_text: str, 
                    qubit: HyperQubit, 
                    vitality: float,
                    tags: List[str] = None):
        """
        Record a new moment in time.
        """
        # Serialize Qubit State
        qs = qubit.state
        state_dict = {
            "alpha": [qs.alpha.real, qs.alpha.imag],
            "beta": [qs.beta.real, qs.beta.imag],
            "gamma": [qs.gamma.real, qs.gamma.imag],
            "delta": [qs.delta.real, qs.delta.imag],
            "w": qs.w, "x": qs.x, "y": qs.y, "z": qs.z
        }
        
        episode = Episode(
            id=str(uuid4()),
            timestamp=time.time(),
            input_text=input_text,
            response_text=response_text,
            qubit_state=state_dict,
            vitality=vitality,
            tags=tags or []
        )
        
        self.episodes.append(episode)
        self._save()
        
        logger.info(f"💾 Memory Recorded: '{input_text[:20]}...' (Vitality: {vitality:.2f})")
        
    def recall_by_resonance(self, current_qubit: HyperQubit, limit: int = 5) -> List[Episode]:
        """
        Find memories that resonate with the *current* state of mind.
        "I feel like this... have I felt this before?"
        """
        if not self.episodes:
            return []
            
        scored_episodes = []
        
        # Current state vector components
        c_qs = current_qubit.state
        
        for ep in self.episodes:
            score = self._calculate_resonance(c_qs, ep.qubit_state)
            scored_episodes.append((score, ep))
            
        # Sort by resonance score (descending)
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        return [ep for score, ep in scored_episodes[:limit]]

    def _calculate_resonance(self, current_state: QubitState, stored_state_dict: Dict[str, Any]) -> float:
        """
        Calculate resonance (similarity) between two quantum states.
        Uses a weighted dot product of the 8D vector components.
        """
        # Reconstruct stored complex numbers
        s_alpha = complex(*stored_state_dict["alpha"])
        s_beta = complex(*stored_state_dict["beta"])
        s_gamma = complex(*stored_state_dict["gamma"])
        s_delta = complex(*stored_state_dict["delta"])
        
        # 1. Amplitude Resonance (Do the bases align?)
        # Point/Line/Space/God alignment
        amp_resonance = (
            abs(current_state.alpha) * abs(s_alpha) +
            abs(current_state.beta) * abs(s_beta) +
            abs(current_state.gamma) * abs(s_gamma) +
            abs(current_state.delta) * abs(s_delta)
        )
        
        # 2. Phase Resonance (Do the spatial orientations align?)
        # W/X/Y/Z alignment
        spatial_dot = (
            current_state.w * stored_state_dict["w"] +
            current_state.x * stored_state_dict["x"] +
            current_state.y * stored_state_dict["y"] +
            current_state.z * stored_state_dict["z"]
        )
        
        # Normalize spatial dot roughly (assuming normalized quaternions)
        # But here w,x,y,z might not be strictly normalized in HyperQubit logic, 
        # so we just take the raw correlation.
        
        # Total Resonance
        return amp_resonance + (spatial_dot * 0.5)

    def get_recent_trajectory(self, n: int = 10) -> List[Episode]:
        """Return the last N episodes to visualize the path."""
        return self.episodes[-n:]

    def _save(self):
        """Save to JSON (Temporary persistence)."""
        try:
            data = [asdict(ep) for ep in self.episodes]
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _load(self):
        """Load from JSON."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.episodes = [Episode(**item) for item in data]
        except FileNotFoundError:
            self.episodes = []
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
