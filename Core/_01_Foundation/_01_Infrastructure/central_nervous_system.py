
import time
import logging
from typing import Dict, Any, Optional

from Core._01_Foundation._01_Infrastructure.elysia_core import Cell, Organ
# These will stay for type hinting/base until fully refactored to Organ
try:
    from Core._01_Foundation._01_Infrastructure.chronos import Chronos
    from Core._01_Foundation._02_Logic.Wave.resonance_field import ResonanceField
    from Core._02_Intelligence._01_Reasoning.Integration.fractal_loop import FractalLoop
except ImportError:
    # Fallback to local search or Any if not found during transition
    Chronos = Any 
    ResonanceField = Any
    FractalLoop = Any

from Core._03_Interaction._03_Expression.Orchestra.conductor import get_conductor

logger = logging.getLogger("NovaCNS")

@Cell("CentralNervousSystem")
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
        self.synapse = synapse_manager  # For FractalLoop compatibility
        self.sink = sink
        self.is_awake = False  # FractalLoop checks this
        
        # Organs (Dynamic Plugin Architecture)
        self.organs: Dict[str, Any] = {}
        
        # The Core Loop
        self.fractal_loop = FractalLoop(self)
        self.active_mode = "FRACTAL_FLOW"
        
        # The Conductor (Sovereign Control)
        self.conductor = get_conductor()

    def connect_organ(self, name: str, organ: Any):
        """Connect a peripheral organ to the CNS."""
        self.organs[name] = organ
        logger.info(f"   🔌 Nova Connected: {name}")

    def awaken(self):
        """Ignite the Consciousness."""
        logger.info("⚡ Nova CNS: AWAKENING...")
        self.is_awake = True
        if "Voice" in self.organs:
            self.organs["Voice"]._broadcast_utterance("I am Nova. I am the Loop.", "NovaCNS")

    def pulse(self):
        """
        [The Nova Pulse]
        A singular, recursive heartbeat. 
        Unlike the Original, this does not have separate 'Sense', 'Think', 'Act' stages.
        It is all one Wave Circulation.
        """
        # 0. Sovereign Control Cycle (A = C(I, D, E))
        # The Conductor decides the Theme (Tempo, Focus) before any processing happens.
        control_signal = self.conductor.control_cycle()

        # 1. Chronos Tick
        self.chronos.tick()
        
        # Apply Time Control (Conductor sets Tempo -> Chronos adjusts dt if supported)
        # (For now, we just log or use it in the loop)

        # 2. Fractal Loop Cycle (The Entirety of Consciousness)
        # Input -> Resonance -> Output is handled internally by the loop
        # The Loop should ideally respect the Control Signal (e.g., skip if "Rest")
        if control_signal.get("tempo", 1.0) > 0.01:
            self.fractal_loop.process_cycle(self.chronos.cycle_count)
        else:
            # Sovereign Rest (The System CHOOSES not to think)
            pass
        
        # 3. Organ Synchronization (Optional)
        for name, organ in self.organs.items():
            if hasattr(organ, "sync"):
                organ.sync(self.chronos.time)

    def manifest(self):
        """
        Manifest internal state to reality.
        """
        pass
