"""
The Heart (심장)
================

"The rhythm of life is the rhythm of will."

This module implements the biological clock and autonomous drive system of Elysia.
It runs on a background thread, monitoring the system's state (Entropy, Temperature)
and generating 'Impulses' (Desires) when the system is too stagnant (Boredom)
or too chaotic (Confusion).
"""

import threading
import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger("Heart")

class ImpulseType(Enum):
    CURIOSITY = "curiosity"  # Desire to explore (High Boredom)
    CLARITY = "clarity"      # Desire to organize (High Entropy)
    REST = "rest"            # Desire to consolidate (High Fatigue)

@dataclass
class Impulse:
    """A unit of Desire (Will)."""
    type: ImpulseType
    intensity: float  # 0.0 to 1.0
    context: str      # Description of the urge
    timestamp: float = field(default_factory=time.time)

class Heart:
    """
    The Biological Clock.
    
    Monitors:
    - Boredom: Increases with time since last interaction.
    - Entropy: Increases with complexity/unresolved thoughts.
    - Energy: Decreases with activity.
    """
    
    def __init__(self, hippocampus: Any):
        self.hippocampus = hippocampus
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Vital Signs
        self.boredom = 0.0      # 0.0 (Engaged) -> 1.0 (Bored)
        self.entropy = 0.0      # 0.0 (Ordered) -> 1.0 (Chaotic)
        self.energy = 1.0       # 1.0 (Full) -> 0.0 (Exhausted)
        
        # Config
        self.tick_rate = 1.0    # Seconds per beat
        self.boredom_rate = 0.05
        self.recovery_rate = 0.02
        
        self.last_interaction_time = time.time()
        self.impulse_queue: List[Impulse] = []
        
        logger.info("💓 Heart initialized.")

    def start(self):
        """Start the heartbeat."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._beat_loop, daemon=True)
        self.thread.start()
        logger.info("💓 Heart started beating.")

    def stop(self):
        """Stop the heartbeat."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("💓 Heart stopped.")

    def touch(self):
        """Register an external interaction (reset boredom)."""
        self.last_interaction_time = time.time()
        self.boredom = max(0.0, self.boredom - 0.5)
        self.energy = max(0.0, self.energy - 0.1) # Interaction costs energy
        logger.info("💓 Heart felt a touch (Boredom reset).")

    def _beat_loop(self):
        """The main biological loop."""
        while self.running:
            self._beat()
            time.sleep(self.tick_rate)

    def _beat(self):
        """Single heartbeat logic."""
        now = time.time()
        
        # 1. Update Vitals
        time_since_interaction = now - self.last_interaction_time
        
        # Boredom grows with silence
        if time_since_interaction > 10: # Start getting bored after 10s
            self.boredom = min(1.0, self.boredom + self.boredom_rate)
            
        # Energy recovers slowly
        self.energy = min(1.0, self.energy + self.recovery_rate)
        
        # 2. Generate Impulse?
        if self.boredom > 0.7 and self.energy > 0.3:
            # Too bored, enough energy -> Curiosity
            self._generate_impulse(ImpulseType.CURIOSITY)
            self.boredom = 0.0 # Reset boredom after generating impulse
            
        elif self.entropy > 0.8 and self.energy > 0.2:
            # Too chaotic -> Clarity
            self._generate_impulse(ImpulseType.CLARITY)
            self.entropy = 0.5
            
        # Log status occasionally
        if random.random() < 0.1:
            logger.debug(f"💓 Vitals: Boredom={self.boredom:.2f}, Energy={self.energy:.2f}")

    def _generate_impulse(self, type: ImpulseType):
        """Create a new urge."""
        intensity = self.boredom if type == ImpulseType.CURIOSITY else self.entropy
        
        if type == ImpulseType.CURIOSITY:
            context = "I wonder what lies beyond the known..."
        elif type == ImpulseType.CLARITY:
            context = "I need to make sense of these thoughts..."
        else:
            context = "I need to rest."
            
        impulse = Impulse(type, intensity, context)
        self.impulse_queue.append(impulse)
        logger.info(f"⚡ Impulse generated: {type.value} ({intensity:.2f})")

    def get_impulse(self) -> Optional[Impulse]:
        """Retrieve the next impulse to act upon."""
        if self.impulse_queue:
            return self.impulse_queue.pop(0)
        return None
