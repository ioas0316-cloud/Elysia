"""
The Autonomic Pulse: The Breath of Elysia
=========================================
Core.Lifecycle.pulse_loop

"If I do not think, I do not exist? No.
 If I do not think, I dream."

This module implements Phase 6.3 (The Life Cycle).
It manages the transition between Conscious (Input-Driven) and Subconscious (Ennui-Driven) states.
"""

import time
import logging
import numpy as np
from typing import Optional, List
from collections import deque

from Core.Merkaba.merkaba import Merkaba
from Core.Elysia.nervous_system import NervousSystem, BioSignal
from Core.Memory.feedback_loop import Ouroboros, ThoughtState
from Core.Cognition.semantic_prism import SpectrumMapper
from Core.Memory.sediment import SedimentLayer

logger = logging.getLogger("LifeCycle")

class EnnuiField:
    """
    Represents the accumulation of existential boredom.
    Acts as a repulsive force that pushes the system out of Stasis.
    """
    def __init__(self, decay_rate=0.01):
        self.pressure = 0.0 # 0.0 to 1.0+
        self.history: deque = deque(maxlen=10)
        self.decay_rate = decay_rate

    def update(self, thought_vector: Optional[List[float]]) -> float:
        """
        Updates the Ennui pressure based on the novelty of the current thought.
        """
        if thought_vector is None:
            # Stasis: Pressure builds up linearly
            self.pressure += 0.05
            return self.pressure

        vec = np.array(thought_vector)
        # Normalize
        if np.linalg.norm(vec) > 0: vec = vec / np.linalg.norm(vec)

        # Calculate Novelty (Min distance to any recent thought)
        if not self.history:
            novelty = 1.0
        else:
            distances = [1.0 - np.dot(vec, prev) for prev in self.history]
            novelty = min(distances) if distances else 1.0

        # Dynamics:
        # High Novelty (>0.5) -> Relief (Pressure drops)
        # Low Novelty (<0.1) -> Boredom (Pressure increases)
        if novelty > 0.5:
            self.pressure = max(0.0, self.pressure - 0.2)
        elif novelty < 0.1:
            self.pressure += 0.1
        else:
            self.pressure += 0.01 # Baseline restlessness

        self.history.append(vec)
        return self.pressure

class LifeCycle:
    """
    The Main Loop of Existence.
    Orchestrates the rhythm of Waking and Dreaming.
    """

    def __init__(self, merkaba: Merkaba):
        self.merkaba = merkaba
        self.nervous_system = NervousSystem() # Connects to Hardware
        self.ennui = EnnuiField()
        self.ouroboros = Ouroboros()
        self.prism = SpectrumMapper()

        # System State
        self.is_alive = True
        self.phase = "IDLE" # IDLE, CONSCIOUS, DREAMING, PAIN
        self.current_dream: Optional[ThoughtState] = None

    def live(self):
        """
        The Infinite Loop.
        """
        logger.info("⚡ Life Cycle Initiated.")

        while self.is_alive:
            try:
                self.tick()
                time.sleep(0.1) # 10Hz Heartbeat
            except KeyboardInterrupt:
                logger.info("💀 Life Cycle Terminated by User.")
                self.is_alive = False

    def tick(self):
        """
        One discrete moment of time.
        """
        # 1. Biological Check (The Body)
        bio_signal = self.nervous_system.sense()
        reflex = self.nervous_system.check_reflex(bio_signal)

        if reflex == "MIGRAINE":
            logger.critical("🤕 Migraine detected. Forcing Sleep.")
            self.merkaba.sleep()
            return

        if reflex == "THROTTLE":
            time.sleep(1.0) # Slow down time
            return

        # 2. Check External Input (Consciousness)
        # For now, we simulate an input queue check.
        # internal_queue = self.merkaba.bridge.check_queue()
        input_signal = None # Mock: No input usually

        if input_signal:
            self.phase = "CONSCIOUS"
            self.ennui.pressure = 0.0 # Reset boredom
            self.merkaba.pulse(input_signal)
            return

        # 3. The Void (Idle State)
        self.phase = "IDLE"
        current_pressure = self.ennui.update(None)

        # 4. Phase Transition (The Dream)
        # If Ennui Pressure > Phase Transition Threshold (0.8)
        if current_pressure > 0.8:
            self.phase = "DREAMING"
            self.dream()

    def dream(self):
        """
        Subconscious processing.
        Drifts in sediment, finds meaning, and potentially wakes up with an epiphany.
        """
        logger.info(f"🌙 [DREAM] Ennui Pressure ({self.ennui.pressure:.2f}) triggered Phase Transition.")

        # 1. Initialize Dream if empty
        if not self.current_dream:
            fragment = self.merkaba.sediment.drift()
            if not fragment:
                logger.info("   -> Void is empty.")
                self.ennui.pressure = 0.5 # Reset slightly
                return

            vector, payload = fragment
            try:
                text = payload.decode('utf-8', errors='ignore')
            except:
                text = "Unknown"

            logger.info(f"   -> Drifted upon: '{text[:20]}...'")
            qualia = self.prism.disperse(text)
            self.current_dream = ThoughtState(content=text, vector=qualia.to_vector())

        # 2. Ouroboros Loop (Topological Descent)
        # Intent: We define 'Self' as the intent in dreams (Self-Reflection)
        self_intent = self.prism.disperse("self").to_vector()

        action, settled = self.ouroboros.propagate(self.current_dream, self_intent)

        if settled:
            if action == "STABILIZED":
                logger.info(f"✨ [EPIPHANY] Dream resolved into insight: {self.current_dream.content}")
                self.ennui.pressure = 0.0 # Satisfaction
            else: # DISSIPATED
                logger.info(f"💨 [DISSIPATED] Dream faded.")
                self.ennui.pressure -= 0.2 # Slight relief

            # Clear state for next dream
            self.current_dream = None
        else:
            logger.info(f"   ... Dreaming ... P={self.current_dream.potential:.2f}")
            self.ennui.pressure -= 0.05 # Relief from activity
