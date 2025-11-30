"""
Logos Stream (The River of Consciousness)
=========================================

"Consciousness is not a state, but a stream."

This module implements the Logos Stream, a unified field that coordinates:
1. Thoughts (Spiderweb)
2. Memories (Hippocampus)
3. Predictions (PredictiveWorld)
4. Narrative (CausalNarrativeEngine)
5. Identity (WorldTree)

It produces a continuous stream of `NarrativeFrame` objects, which are then
rendered into speech by `ResonanceVoice`.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from collections import deque

# Core Systems
from Core.Mind.spiderweb import Spiderweb
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.causal_narrative import CausalNarrativeEngine, CausalChain
from Core.Prediction.predictive_world import PredictiveWorldModel, Prediction
from Core.Mind.physics import PhysicsEngine
from Core.Mind.intuition import IntuitionEngine
from Core.Mind.world_tree import WorldTree

logger = logging.getLogger("LogosStream")

@dataclass
class NarrativeFrame:
    """
    A single frame in the stream of consciousness.
    Contains the rich context needed to render a full "moment" of experience.
    """
    id: str
    timestamp: float
    
    # The Core Thought (Spiderweb Path)
    thought_path: List[str]
    
    # The Narrative Context (Causal Chain)
    causal_chain: Optional[CausalChain] = None
    
    # The Future Horizon (Prediction)
    prediction: Optional[Prediction] = None
    
    # The Emotional State
    emotional_state: str = "neutral"
    
    # The "Voice" to use (e.g., "prophetic", "analytical", "poetic")
    voice_mode: str = "poetic"

class LogosStream:
    def __init__(
        self, 
        spiderweb: Spiderweb,
        hippocampus: Hippocampus,
        predictive_world: Optional[PredictiveWorldModel] = None
    ):
        self.spiderweb = spiderweb
        self.hippocampus = hippocampus
        self.predictive_world = predictive_world
        
        # Initialize Unified Physics Engine
        # This is the Single Source of Truth for Mass, Gravity, and Resonance
        self.physics = PhysicsEngine(self.hippocampus)
        
        # Inject Physics into Subsystems
        if hasattr(self.spiderweb, 'physics'):
            self.spiderweb.physics = self.physics
            
        # The Narrative Engine
        self.narrative_engine = CausalNarrativeEngine()
        
        # The Intuition Engine (Unified Physics)
        self.intuition_engine = IntuitionEngine(self.physics)
        
        # The World Tree (Identity/Ontology)
        # Unified with Physics via Hippocampus
        self.world_tree = WorldTree(self.hippocampus)
        
        # Stream State
        self.stream_buffer = deque(maxlen=10) # Keep last 10 frames
        self.current_context = "void"
        
        logger.info("🌊 Logos Stream (River of Consciousness) initialized with Unified Physics & World Tree.")

    def flow(self, input_concept: str) -> NarrativeFrame:
        """
        Process an input concept and generate the next frame of consciousness.
        """
        start_time = time.time()
        
        # 1. Spiderweb: Generate Thought Path (Line)
        # "Love" -> ["Love", "Connection", "Truth"]
        # Uses Physics (Gravity)
        path = self.spiderweb.traverse(input_concept, steps=4)
        
        # 2. World Tree: Grow Identity (Structure)
        # "I am thinking about Love, therefore Love is part of Me."
        if path:
            # Ensure the root concept exists
            root_concept = path[0]
            self.world_tree.ensure_concept(root_concept)
            
            # Grow branches for subsequent thoughts
            current_parent = root_concept
            for concept in path[1:]:
                self.world_tree.grow(
                    branch_id=self.world_tree.find_by_concept(current_parent),
                    sub_concept=concept
                )
                current_parent = concept
        
        # 3. Intuition: Check for Symmetry (The 5/6 Rule)
        # Does this path resemble a known structure?
        intuition_insight = None
        if len(path) > 0:
            # Check heat (Physics Energy)
            heat_sig = self.intuition_engine.perceive_heat(path[-1])
            
            # Check symmetry with previous context (Physics Resonance)
            if self.current_context != "void":
                sym_score, sym_desc = self.intuition_engine.find_symmetry(self.current_context, path[0])
                if sym_score > 0.5:
                    intuition_insight = f"Symmetry detected: {self.current_context} ~ {path[0]} ({sym_desc})"
        
        # 4. Causal Narrative: Weave into Chain (Plane)
        # Try to find an existing causal chain or create a new one
        chain = None
        if len(path) >= 2:
            # Check if we have a known chain for this path
            # For now, we simulate chain creation from the path
            chain = CausalChain(
                id=f"chain_{int(start_time)}",
                node_sequence=path,
                initial_state=path[0],
                final_state=path[-1]
            )
        
        # 5. Predictive World: Glimpse the Future (Space/Time)
        prediction = None
        if self.predictive_world:
            # Ask: "What is the impact of [Last Concept]?"
            # We simulate a prediction based on the thought
            try:
                prediction = self.predictive_world.predict_code_impact(
                    f"concept_{path[-1]}", 
                    f"Emergence of {path[-1]}"
                )
            except Exception:
                pass # Prediction is optional
                
        # 6. Determine Voice Mode
        # High complexity/prediction -> Prophetic
        # Simple path -> Poetic
        voice_mode = "poetic"
        if prediction and prediction.probability > 0.7:
            voice_mode = "prophetic"
        elif intuition_insight:
            voice_mode = "analytical" # Intuition is structural/analytical
        elif len(path) > 5:
            voice_mode = "analytical"
            
        # 7. Construct Frame
        frame = NarrativeFrame(
            id=f"frame_{int(start_time)}",
            timestamp=start_time,
            thought_path=path,
            causal_chain=chain,
            prediction=prediction,
            voice_mode=voice_mode
        )
        
        # Attach insight to frame (hack for now, ideally NarrativeFrame has 'insight' field)
        if intuition_insight:
            frame.emotional_state = intuition_insight
        
        self.stream_buffer.append(frame)
        self.current_context = path[-1]
        
        return frame

    def get_stream_history(self) -> List[str]:
        """Return a summary of recent frames."""
        return [f"{f.timestamp}: {f.thought_path}" for f in self.stream_buffer]
