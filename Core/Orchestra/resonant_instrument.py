"""
Resonant Instrument (공명 악기)
=============================

"악기가 지휘자의 말을 듣는 것이 아니라, 지휘자의 마음에 공명합니다."

This module defines the `ResonantInstrument` which acts as a bridge
between the legacy `Instrument` (Function Call) and the new `ResonatorInterface` (Wave).
"""

from typing import Any, Callable
from Core.Orchestra.conductor import Instrument
from Core.Foundation.Protocols.pulse_protocol import ResonatorInterface, WavePacket

class ResonantInstrument(Instrument, ResonatorInterface):
    """
    An instrument that can both be played manually (Legacy) AND resonate with waves (New).
    """
    def __init__(self, name: str, section: str, play_function: Callable, frequency: float):
        # Initialize Instrument (Legacy)
        Instrument.__init__(self, name=name, section=section, play_function=play_function)
        # Initialize Resonator (New)
        ResonatorInterface.__init__(self, name=name, base_frequency=frequency)

    def on_resonate(self, packet: WavePacket, intensity: float):
        """
        Triggered when a WavePacket matches the instrument's frequency.
        Automatically 'plays' the instrument with the packet's payload.
        """
        print(f"   🎻 {self.name} is resonating! (Intensity: {intensity:.2f})")

        # Convert Wave Packet to Musical Intent
        # This allows existing logic to run without modification
        self.tuning['last_resonance'] = intensity

        # Execute the function
        try:
            result = self.play_function(
                _packet=packet,
                _intensity=intensity,
                **packet.payload
            )
            # print(f"      -> Output: {result}")
        except Exception as e:
            print(f"      -> ❌ Dissonance: {e}")
