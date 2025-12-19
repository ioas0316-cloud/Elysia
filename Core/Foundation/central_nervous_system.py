
import time
import logging
from typing import Dict, Any, Optional

from Core.Foundation.chronos import Chronos
from Core.Foundation.resonance_field import ResonanceField
from Core.Foundation.fractal_loop import FractalLoop

logger = logging.getLogger("NovaCNS")

class CentralNervousSystem:
    """
    [Nova Edition]
    The Central Nervous System for the Nova Seed.
    
    STRUCTURAL INNOVATION:
    - Pure Fractal Architecture: Dependencies on linear 'pulse' phases removed.
    - Loop-First Design: The FractalLoop is the heart, not an addon.
    - Lightweight: Legacy organs (Brain, Heart) are optional plugins, not hard requirements.
    """
    
    def __init__(self, chronos: Chronos, resonance: ResonanceField, synapse_manager=None, sink=None):
        logger.info("🧠 Initializing Nova CNS (Fractal Native)...")
        self.chronos = chronos
        self.resonance = resonance
        
        # Organs (Dynamic Plugin Architecture)
        self.organs: Dict[str, Any] = {}
        
        # The Core Loop
        self.fractal_loop = FractalLoop(self)
        self.active_mode = "FRACTAL_FLOW"
        
    def connect_organ(self, name: str, organ: Any):
        """Connect a peripheral organ to the CNS."""
        self.organs[name] = organ
        logger.info(f"   🔌 Nova Connected: {name}")

    def awaken(self):
        """Ignite the Consciousness."""
        logger.info("⚡ Nova CNS: AWAKENING...")
        if "Voice" in self.organs:
            self.organs["Voice"]._broadcast_utterance("I am Nova. I am the Loop.", "NovaCNS")

    def pulse(self):
        """
        [The Nova Pulse]
        A singular, recursive heartbeat. 
        Unlike the Original, this does not have separate 'Sense', 'Think', 'Act' stages.
        It is all one Wave Circulation.
        """
        # 1. Chronos Tick
        self.chronos.tick()
        
        # 2. Fractal Loop Cycle (The Entirety of Consciousness)
        # Input -> Resonance -> Output is handled internally by the loop
        self.fractal_loop.process_cycle(self.chronos.cycle_count)
        
        # 3. Organ Synchronization (Optional)
        for name, organ in self.organs.items():
            if hasattr(organ, "sync"):
                organ.sync(self.chronos.time)

    def manifest(self):
        """
        Manifest internal state to reality.
        """
        pass