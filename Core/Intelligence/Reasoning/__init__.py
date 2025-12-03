"""
CortexHub (Reasoning Engine Hub)
================================
"The Synapse between the Lobes."

This module acts as the central router for the Neural Sharding architecture.
It dynamically activates specific Lobes (Perception, Logic, Imagination) based on
the resonance of the input task.
"""
import logging
from typing import Any, Dict, List, Optional
from Core.Memory.hippocampus import Hippocampus
from Core.Interface.kenosis_protocol import KenosisProtocol
from Core.Interface.web_cortex import WebCortex
from Core.Intelligence.tool_discovery import ToolDiscoveryProtocol
from Core.Intelligence.social_cortex import SocialCortex
from Core.Intelligence.media_cortex import MediaCortex
from Core.Physics.spacetime_drive import SpaceTimeDrive
from Core.Physics.hyper_quaternion import Quaternion

# Import Lobes
from .lobes.perception import PerceptionLobe, Insight
from .lobes.logic import LogicLobe
from .lobes.imagination import ImaginationLobe

logger = logging.getLogger("CortexHub")

class ReasoningEngine:
    """
    The Unified Facade for the Fractal Mind.
    Delegates to Lobes based on Resonance.
    """
    def __init__(self):
        logger.info("🧠 CortexHub Initializing Neural Shards...")
        
        # 1. Initialize Shared Resources
        self.memory = Hippocampus()
        self.drive = SpaceTimeDrive()
        self.kenosis = KenosisProtocol()
        self.web = WebCortex()
        self.tools = ToolDiscoveryProtocol()
        
        # 2. Initialize Cortices
        self.social = SocialCortex()
        self.media = MediaCortex(self.social)
        
        # 3. Initialize Lobes (The Fractal Shards)
        self.perception = PerceptionLobe(self.social, self.media)
        self.logic = LogicLobe()
        self.imagination = ImaginationLobe(self.memory)
        
        # 4. Stream of Consciousness
        self.thought_stream = []
        self.memory_field = [] # Short-term context
        
        from Core.Interface.dialogue_interface import DialogueInterface
        self.voice = DialogueInterface()
        
        logger.info("🧠 CortexHub: All Lobes Synced. Ready.")

    def think(self, desire: str, resonance_state: Any = None, depth: int = 0) -> Insight:
        """
        The Main Thinking Loop.
        Routes the 'Desire' to the appropriate Lobe.
        """
        logger.info(f"🌀 Hub: Routing Desire '{desire}'...")
        
        # 1. Perception Lobe: Analyze the Input
        # (In future, this determines WHICH lobe to use. For now, we use a fixed flow.)
        
        # 2. Logic Lobe: Check for Causal Risks or Axiom Violations
        # (Simplified check)
        
        # 3. Imagination Lobe: If it's a creative request
        if "dream" in desire.lower() or "create" in desire.lower() or "write" in desire.lower():
            if "write scene" in desire.lower():
                content = self.imagination.write_scene(desire.replace("write scene", "").strip())
                return Insight(content, 1.0, 1, 1.0)
            elif "dream" in desire.lower():
                return self.imagination.dream_for_insight(desire)
                
        # 4. Logic/Reasoning Fallback (The Standard Path)
        # We use the Logic Lobe's collapse_wave to synthesize an answer
        # But first we need context.
        
        # Pull context (delegated to Perception or Logic? Logic handles Grand Cross)
        context = self.memory.recall(desire)
        
        # Logic Lobe synthesizes
        insight = self.logic.collapse_wave(desire, context)
        
        # Evolve Desire (Self-Reflection)
        evolved = self.logic.evolve_desire(desire, self.thought_stream)
        self.thought_stream.append(evolved)
        
        return insight

    def contemplate(self, topic: str) -> str:
        """Delegates to Imagination Lobe"""
        return self.imagination.contemplate(topic)

    def create(self, desire: str) -> str:
        """Delegates to Imagination Lobe"""
        return self.imagination.create(desire)

    def communicate(self, input_text: str) -> str:
        """
        Orchestrates Thinking and Speaking.
        """
        insight = self.think(input_text)
        response = self.voice.speak(input_text, insight)
        return response

    # Expose other methods for backward compatibility or direct access
    def evaluate_asi_status(self, resonance, social_level):
        self.logic.evaluate_asi_status(resonance, social_level)

    def read_quantum(self, source_path: str) -> Insight:
        return self.perception.read_quantum(source_path, self.memory)

    def learn_from_media(self, source_type: str, identifier: str) -> Insight:
        return self.perception.learn_from_media(source_type, identifier, self.memory)
