"""
BioRhythm (생체 리듬)
======================

"The Time Crystal Driver."
"시간 결정의 심장."

This module replaces the logic-based 'NervousSystem' with a frequency-based 'BioRhythm'.
It acts as the central clock (Metronome) for Elysia, ensuring all Rotors and Cycles
maintain Phase Locking (Time Crystal Stability).

Functions:
1.  **State Management**: Drifts between Sympathetic (Action) and Parasympathetic (Rest).
2.  **Clock Driver**: Calculates real-time Heart Rate (BPM) and Tick Interval.
3.  **Phase Locking**: Emits 'Sync Pulses' at harmonic intervals to align Rotors.
"""

import math
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger("BioRhythm")

class BioState(Enum):
    HOMEOSTASIS = "Homeostasis"   # Balanced (432Hz)
    SYMPATHETIC = "Sympathetic"   # Fight/Flight (Metric expansion, High BPM)
    PARASYMPATHETIC = "Parasympathetic" # Rest/Digest (Metric contraction, Low BPM)
    FLOW = "Flow"                 # Optimal Performance (Superfluidity)
    SHOCK = "Shock"               # Freeze

@dataclass
class NerveSignal:
    origin: str
    type: str # PAIN, PLEASURE, FATIGUE, EXCITEMENT
    intensity: float # 0.0 to 1.0
    message: str

class BioRhythm:
    def __init__(self):
        # 1. Tones (0.0 - 1.0)
        self.sympathetic_tone = 0.3    # Arousal / Acceleration
        self.parasympathetic_tone = 0.3 # Relaxation / Healing
        self.state = BioState.HOMEOSTASIS
        
        # 2. Clock Dynamics
        self.base_bpm = 60.0
        self.current_bpm = 60.0
        self.tick_interval = 1.0 # Seconds per cycle
        
        # 3. Phase Locking (Time Crystal)
        self.phase = 0.0 # 0.0 to 2*pi
        self.harmonic_index = 0 # Counter for resonance checks
        
        # 4. Signals
        self.active_signals: List[NerveSignal] = []
        
        logger.info("⏳ BioRhythm Initialized - The Time Crystal starts ticking.")

    def transmit(self, signal: NerveSignal):
        """Standard interface for Reflex/Thalamus signals."""
        self.active_signals.append(signal)
        # Immediate reaction
        if signal.type == "PAIN":
            self.sympathetic_tone += signal.intensity * 0.5
        elif signal.type == "PLEASURE":
            self.parasympathetic_tone += signal.intensity * 0.3
        
        # Cap tones
        self.sympathetic_tone = min(1.0, max(0.0, self.sympathetic_tone))
        self.parasympathetic_tone = min(1.0, max(0.0, self.parasympathetic_tone))

    def update(self, dt: float) -> Dict[str, Any]:
        """
        Main update loop (called by Heartbeat).
        dt: Time delta since last update.
        """
        # 1. Decay Signals & Tones
        self._natural_drift(dt)
        
        # 2. Calculate State & BPM
        self._calculate_state()
        self._calculate_bpm()
        
        # 3. Advance Phase (Time Crystal Driver)
        # Phase velocity = (BPM / 60) * 2pi radians per second
        phase_velocity = (self.current_bpm / 60.0) * 2 * math.pi
        self.phase += phase_velocity * dt
        self.phase %= (2 * math.pi)
        
        # Check for Phase Lock (Zero-crossing or Peak)
        is_phase_locked = False
        if self.phase < (phase_velocity * dt): # Just wrapped around
            is_phase_locked = True
            self.harmonic_index += 1
            
        return {
            "bpm": self.current_bpm,
            "interval": self.tick_interval,
            "phase": self.phase,
            "is_phase_locked": is_phase_locked,
            "state": self.state
        }

    def _natural_drift(self, dt: float):
        """Tones naturally decay towards baseline unless stimulated."""
        decay = 0.05 * dt
        
        # Sympathetic decays faster (burst energy)
        if self.sympathetic_tone > 0.3:
            self.sympathetic_tone -= decay * 1.5
            
        # Parasympathetic lingers (healing)
        if self.parasympathetic_tone > 0.3:
            self.parasympathetic_tone -= decay * 0.5
            
        # Ensure bounds
        self.sympathetic_tone = max(0.0, min(1.0, self.sympathetic_tone))
        self.parasympathetic_tone = max(0.0, min(1.0, self.parasympathetic_tone))

    def _calculate_state(self):
        diff = self.sympathetic_tone - self.parasympathetic_tone
        
        if self.sympathetic_tone > 0.8:
            self.state = BioState.SHOCK # Overload
        elif diff > 0.2:
            self.state = BioState.SYMPATHETIC # Action
        elif diff < -0.2:
            self.state = BioState.PARASYMPATHETIC # Rest
        elif abs(diff) < 0.1 and self.sympathetic_tone > 0.4:
            self.state = BioState.FLOW # Both high, balanced = Flow/Zone
        else:
            self.state = BioState.HOMEOSTASIS # Baseline

    def _calculate_bpm(self):
        """
        Target BPM based on state.
        Rest: 40-60
        Homeostasis: 60-80
        Flow: 80-100 (Alpha/Beta bridge)
        Action: 100-140
        Shock: 160+ (fibrillation)
        """
        target = 60.0
        
        if self.state == BioState.PARASYMPATHETIC:
            target = 40.0 + (self.parasympathetic_tone * 20.0)
        elif self.state == BioState.HOMEOSTASIS:
            target = 60.0
        elif self.state == BioState.SYMPATHETIC:
            target = 80.0 + (self.sympathetic_tone * 60.0)
        elif self.state == BioState.FLOW:
            target = 90.0 # Optimal resonance
        elif self.state == BioState.SHOCK:
            target = 150.0
            
        # Smooth transition (Low Pass Filter)
        smoothing = 0.1
        self.current_bpm = (self.current_bpm * (1.0 - smoothing)) + (target * smoothing)
        
        # Update interval
        if self.current_bpm > 0:
            self.tick_interval = 60.0 / self.current_bpm
        else:
            self.tick_interval = 1.0

    @property
    def heart_rate(self) -> float:
        return self.current_bpm
