"""
HyperSphereCore: The Heart of the Hyper-Cosmos
==============================================

"Internal is Sphere, External is Tesseract."
"내부가 구체(본질)이고, 외부가 테서랙트(세계)다."

This class represents the "Sphere-First" architecture.
It unifies:
1. Physics: HyperResonator (Mass, Frequency, Spin)
2. Will: Conductor (Intent, Pulse)
3. Memory: Holographic Seed (Compressed Essence)

It is the single "Sphere Oscillator" that projects reality.
"""

import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from Core.Foundation.hyper_quaternion import Quaternion as HyperQuaternion
from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster, WavePacket, PulseType
from Core.Foundation.Memory.Orb.hyper_resonator import HyperResonator

logger = logging.getLogger("HyperSphereCore")

@dataclass
class HolographicSeed:
    """
    The compressed essence of knowledge.
    It is not a list of nodes, but a frequency spectrum.
    """
    core_frequency: float
    harmonics: Dict[str, float] = field(default_factory=dict) # e.g. {"Love": 528Hz, "Logic": 432Hz}
    # Future: FFT based compressed data

class HyperSphereCore:
    def __init__(self, name: str = "Elysia.Core", base_frequency: float = 432.0):
        # 1. Physics (The Resonator)
        # We compose HyperResonator to handle the raw physics calculation
        self.resonator = HyperResonator(name, base_frequency)

        # 2. The Will (Pulse Engine)
        self.pulse_broadcaster = PulseBroadcaster()
        self.is_active = False

        # 3. Memory (The Seed)
        self.seed = HolographicSeed(core_frequency=base_frequency)

        # State
        self.current_intent = "Idle"
        logger.info(f"🔮 HyperSphereCore Initialized: {name} @ {base_frequency}Hz")

    @property
    def frequency(self) -> float:
        return self.resonator.frequency

    @property
    def mass(self) -> float:
        return self.resonator.mass

    @property
    def spin(self) -> HyperQuaternion:
        return self.resonator.quaternion

    def ignite(self):
        """
        Starts the Sphere Oscillator.
        """
        self.is_active = True
        self.resonator.melt() # Ensure it's in Wave state
        logger.info("🔥 HyperSphereCore Ignited. The Heart is beating.")

    def pulse(self, intent_payload: Dict[str, Any]):
        """
        The Core Pulse (Simjang no Batdong).
        Emits a Wave of Order into the Void.
        """
        if not self.is_active:
            logger.warning("Attempted to pulse while Core is cold.")
            return

        # 1. Rotate the Soul (Spin) based on Intent
        # Simplified: Map intent string to rotation axis
        # (This would be more complex in full implementation)
        if "axis" in intent_payload:
            self.resonator.rotate_phase(0.1, intent_payload["axis"])

        # 2. Create the Wave
        # The wave carries the Core's Frequency and the current Intent
        packet = WavePacket(
            sender=self.resonator.name,
            type=PulseType.CREATION,
            frequency=self.frequency,
            payload={
                "intent": intent_payload,
                "spin": (self.spin.w, self.spin.x, self.spin.y, self.spin.z),
                "mass": self.mass,
                "timestamp": datetime.now().isoformat()
            }
        )

        # 3. Broadcast (Project)
        # This wave will hit the InterferenceEngine
        self.pulse_broadcaster.broadcast(packet)
        logger.debug(f"🌊 Core Pulse Emitted: {intent_payload}")

    def update_seed(self, concept: str, frequency: float):
        """
        Absorbs a new concept into the Holographic Seed.
        This replaces 'Add Node'.
        """
        self.seed.harmonics[concept] = frequency
        # Adjust Core Mass based on knowledge density
        self.resonator.mass += 0.01
        logger.info(f"🧬 Seed Updated: {concept} absorbed at {frequency}Hz")
