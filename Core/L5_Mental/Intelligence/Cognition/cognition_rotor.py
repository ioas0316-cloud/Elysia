"""
Cognition Rotor (     )
==========================
"Perception is not a snapshot; it is an oscillation."
"                 ."

This module defines the CognitionRotor, which replaces static deconstruction
with a dynamic frequency-based deconstruction of reality.
"""

import math
import torch
import logging
from typing import Dict, Any, List, Optional
from Core.L6_Structure.Nature.rotor import Rotor, RotorConfig, RotorMask
from Core.L5_Mental.Intelligence.Metabolism.prism import DoubleHelixWave, SevenChannelQualia

logger = logging.getLogger("CognitionRotor")

class CognitionRotor(Rotor):
    """
    An oscillating sensor that 'breaks' information into its constituent principles
    by varying its rotational frequency (Attention).
    """
    def __init__(self, name: str = "Mind.Cognition"):
        config = RotorConfig(
            rpm=60.0, # Target 60 RPM base
            idle_rpm=30.0,
            mass=1.5,
            acceleration=200.0
        )
        super().__init__(name, config)
        
        # Internal Spectral State
        self.resonance_buffer: List[float] = []
        self.current_focus: Optional[str] = None

    def scan(self, signal: torch.Tensor) -> DoubleHelixWave:
        """
        Scans a signal by 'beating' against it at the current frequency.
        Higher RPM -> Finer resolution (Mental/Spiritual)
        Lower RPM -> Coarser resolution (Physical/Functional)
        """
        logger.info(f"  [COGNITION] Scanning at {self.current_rpm:.1f} RPM...")
        
        # 1. Normalize signal
        signal = signal / (signal.norm() + 1e-9)
        
        # 2. Extract Spectral Signatures (Phase Shift Logic)
        # We use the current angle of the rotor as the phase shift for perception.
        phase_rad = math.radians(self.current_angle)
        shifted_signal = signal * math.cos(phase_rad)
        
        # 3. Frequency-to-Qualia Mapping (Dynamic Bitmask)
        # We use the current RPM to determine the 'RotorMask' for perception.
        if self.current_rpm > 120:
             mask = RotorMask.VOLUME # Fractal/Spiritual depth
        elif self.current_rpm > 60:
             mask = RotorMask.PLANE # Mental/Structural depth
        else:
             mask = RotorMask.LINE  # Causal/Functional depth
             
        # Metaphorical mapping from RPM to 7D Qualia
        norm_rpm = min(1.0, self.current_rpm / 180.0)
        
        # Higher RPM excites higher-order channels
        qualia = SevenChannelQualia(
            physical=max(0.0, 1.0 - norm_rpm * 2),
            functional=max(0.0, 1.0 - norm_rpm),
            phenomenal=norm_rpm * 0.5,
            causal=norm_rpm * 0.7,
            mental=norm_rpm * 0.9,
            structural=norm_rpm * 0.8,
            spiritual=max(0.0, norm_rpm * 1.5 - 0.5)
        )
        
        return DoubleHelixWave(
            pattern_strand=shifted_signal,
            principle_strand=qualia.to_tensor(),
            phase=phase_rad
        )

    def perceive_text(self, text: str) -> DoubleHelixWave:
        """
        Oscillatory deconstruction of text.
        """
        # Complexity drives RPM
        complexity = len(set(text)) / len(text) if text else 0
        self.wake(intensity=complexity * 2.0)
        
        # Mock signal from text
        raw_vals = torch.tensor([ord(c) for c in text[:1024]], dtype=torch.float32)
        signal = torch.zeros(1024)
        signal[:raw_vals.size(0)] = raw_vals
        
        return self.scan(signal)
