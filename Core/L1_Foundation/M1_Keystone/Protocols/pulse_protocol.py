"""
Pulse Protocol (       )
============================

"The heartbeat of the system. Listen, and you shall act."

      Phase 1: The Pulse             .
   (Conductor)     (Resonator/Module)                .

    `Function Call`          `Wave Broadcast`          .
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

class PulseType(Enum):
    """
           (Signal Type)
    """
    SYNCHRONIZATION = "Sync"       #        (Heartbeat)
    INTENTION_SHIFT = "Shift"      #       (Mode Change)
    ATTENTION_FOCUS = "Focus"      #          (Gravity Boost)
    RELAXATION = "Relax"           #         (Entropy Decay)
    EMERGENCY = "Emergency"        #       (High Alert)
    CREATION = "Genesis"           #       (Creative Burst)
    SENSORY = "Sensory"            #       (Input)
    KNOWLEDGE = "Knowledge"        #       (Learning)
    MEMORY_STORE = "Store"         #       (Freeze)
    MEMORY_RECALL = "Recall"       #       (Melt)
    DREAM_CYCLE = "Dream"          #   (Sleep Cycle)

@dataclass
class WavePacket:
    """
          (The Signal)

                       .
    """
    sender: str               #     (Source)
    type: PulseType           # (Renamed from pulse_type for brevity, aliased below)
    frequency: float = 432.0  # Hz (Target Domain: 400=Body, 500=Mind, 600=Spirit)
    amplitude: float = 1.0    # 0.0 ~ 1.0 (Intensity/Priority)
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict) #        (Context)

    # Backwards compatibility alias
    @property
    def pulse_type(self): return self.type

    @property
    def intensity(self): return self.amplitude

    @property
    def energy(self) -> float:
        """E = hf * A (Energy proportional to Frequency * Amplitude)"""
        return self.frequency * self.amplitude

class ResonatorInterface:
    """
              (The Listener)

                                     .
    """
    def __init__(self, name: str, base_frequency: float):
        self.name = name
        self.base_frequency = base_frequency #        (Resonance Frequency)
        self.current_vibration = 0.0

    def listen(self, packet: WavePacket) -> bool:
        """
                         .

        Returns:
            True if resonated (Active), False if ignored (Dormant).
        """
        # 1. Frequency Matching (Resonance Logic)
        #                 (Bandwidth:  50Hz)
        diff = abs(packet.frequency - self.base_frequency)
        if diff < 50.0:
            resonance_factor = (50.0 - diff) / 50.0 # 1.0 (Exact) -> 0.0 (Edge)
            self.on_resonate(packet, resonance_factor * packet.amplitude)
            return True
        return False

    def on_resonate(self, packet: WavePacket, intensity: float):
        """
                       . (코드 베이스 구조 로터)
        """
        raise NotImplementedError("Resonators must implement on_resonate()")

class PulseBroadcaster:
    """
           (The Heart/Conductor's Mouth)
    """
    def __init__(self):
        self.listeners: List[ResonatorInterface] = []

    def register(self, listener: ResonatorInterface):
        self.listeners.append(listener)

    def broadcast(self, packet: WavePacket):
        """
                          .
        """
        # Hook for Traffic Controller (The City Monitor)
        try:
            # Lazy import to avoid circular dependency
            from Core.Scripts.traffic_controller import get_traffic_controller
            get_traffic_controller().on_resonate(packet, packet.amplitude)
        except ImportError:
            pass # Monitor not available, ignore

        # TODO:         (Async)          .
        active_count = 0
        for listener in self.listeners:
            if listener.listen(packet):
                active_count += 1
        return active_count
