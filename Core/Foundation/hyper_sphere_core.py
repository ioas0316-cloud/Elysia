"""
HyperSphereCore: The Heart of the Hyper-Cosmos
==============================================

"The Rotor creates the Sphere."
"로터가 구체를 만든다."

This class represents the "Sphere-First" architecture.
It unifies:
1. Physics: Rotor System (Oscillators)
2. Will: Conductor (Intent)
3. Memory: Holographic Seed (Rotor Configuration)

Philosophy:
- "Make it breathe. Make it spin."
- The Sphere is a living engine that breathes (Low RPM) when idle
  and pulses (High RPM) when active.
"""

import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from Core.Foundation.hyper_quaternion import Quaternion as HyperQuaternion
from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster, WavePacket, PulseType
from Core.Foundation.Nature.rotor import Rotor, RotorConfig

logger = logging.getLogger("HyperSphereCore")

class HyperSphereCore:
    def __init__(self, name: str = "Elysia.Core", base_frequency: float = 432.0):
        self.name = name

        # --- THE ENGINE: ROTOR SYSTEM ---
        # The Core is not a single point, but a collection of spinning Rotors.
        # Primary Rotor (The Self)
        self.primary_rotor = Rotor(
            name="Self",
            config=RotorConfig(rpm=base_frequency * 60, mass=100.0) # Base Freq -> RPM
        )

        # Harmonic Rotors (The Knowledge/Seed)
        # Instead of a static dict, we have Rotors spinning at different speeds.
        self.harmonic_rotors: Dict[str, Rotor] = {}

        # --------------------------------

        self.pulse_broadcaster = PulseBroadcaster()
        self.is_active = False

        # State
        self.current_intent = "Sleep"
        logger.info(f"🔮 HyperSphereCore (Rotor Engine) Initialized: {name}")

    @property
    def frequency(self) -> float:
        return self.primary_rotor.frequency_hz

    @property
    def mass(self) -> float:
        return self.primary_rotor.config.mass

    @property
    def spin(self) -> HyperQuaternion:
        # Map Rotor Phase to Quaternion (Manual Implementation)
        # q = cos(theta/2) + (u * sin(theta/2))
        # Axis u is Z-axis (0, 0, 1)
        theta = math.radians(self.primary_rotor.current_angle)
        half_theta = theta / 2.0

        w = math.cos(half_theta)
        z = math.sin(half_theta) # x, y are 0 since axis is (0,0,1)

        return HyperQuaternion(w, 0, 0, z)

    def ignite(self):
        """
        Starts the Rotor Engine (Wakes up).
        """
        self.is_active = True
        self.primary_rotor.spin_up()
        for r in self.harmonic_rotors.values():
            r.spin_up()
        logger.info("🔥 HyperSphereCore Ignited. Rotors are spinning (Awake).")

    def pulse(self, intent_payload: Dict[str, Any], dt: float = 1.0):
        """
        The Core Pulse.
        1. Advances Rotors (Time Step).
        2. Calculates Superposition (Wave).
        3. Broadcasts.
        """
        if not self.is_active:
            # In a real system, we'd call update() here too to ensure "Breathing"
            # even if 'ignite' hasn't been called fully.
            # But ignite() sets is_active=True, so this block handles "Off" state.
            pass

        # 1. Update Physics (Advance Phase & Breath)
        self.primary_rotor.update(dt)
        harmonics_snapshot = {}

        for name, rotor in self.harmonic_rotors.items():
            rotor.update(dt)
            freq, amp, phase = rotor.get_wave_component()
            harmonics_snapshot[name] = freq
            # Note: We pass freq for Interference Engine to match.
            # Ideally, we'd pass the full (freq, phase) for complex interference.

        # 2. Create the Wave
        # The wave carries the Superposition State of all rotors
        packet = WavePacket(
            sender=self.name,
            type=PulseType.CREATION,
            frequency=self.frequency,
            payload={
                "intent": {
                    "harmonics": harmonics_snapshot, # The Spectrum
                    "payload": intent_payload
                },
                "spin": (self.spin.w, self.spin.x, self.spin.y, self.spin.z),
                "mass": self.mass,
                "timestamp": datetime.now().isoformat(),
                "phase": self.primary_rotor.current_angle # Debug info
            }
        )

        # 3. Broadcast
        self.pulse_broadcaster.broadcast(packet)
        # logger.debug(f"🌊 Core Pulse: {self.primary_rotor}")

    def update_seed(self, concept: str, frequency: float):
        """
        Adds a new Rotor for the concept.
        """
        rotor = Rotor(
            name=concept,
            config=RotorConfig(rpm=frequency * 60, mass=10.0)
        )
        if self.is_active:
            rotor.spin_up()

        self.harmonic_rotors[concept] = rotor
        self.primary_rotor.config.mass += 0.1 # Self gains mass from knowledge
        logger.info(f"🧬 Seed Updated (Rotor Added): {rotor}")
