"""
Dream Rotor (The Geometry of Reverie)
=====================================
Core.System.dream_rotor

"Dreams are not movies; they are spinning worlds."

This module defines the DreamRotor, a physics object that represents
a unit of dream processing. It adds physicality (Spin, Tilt, Tension)
to the abstract data of a dream.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DreamRotor:
    intent: str
    vector_dna: List[float]
    void_distance: float = 0.0
    
    # Physics State
    rpm: float = 0.0          # Spin Speed (Emotional Intensity)
    tilt_angle: float = 0.0   # Degrees from Reality Axis (0.0 = Reality)
    tension: float = 0.0      # Elastic Force pulling back to Center
    stability: float = 1.0    # 1.0 = Stable, 0.0 = Collapse
    polarity: float = 0.0     # -1.0 (Pain) to +1.0 (Joy/Flow)
    
    def __post_init__(self):
        # Initial Physics Calculation
        self._calculate_geometry()

    def _calculate_geometry(self):
        """Calculates initial geometry based on vector DNA."""
        # Distance (Magnitude) is the primary driver for Tilt
        magnitude = math.sqrt(sum(x**2 for x in self.vector_dna))
        
        # If void_distance wasn't provided, use magnitude
        if self.void_distance == 0.0:
            self.void_distance = magnitude
            
        # Physics Constants
        # TILT_FACTOR: How much distance converts to Tilt?
        # Safe Orbit (3.0) should be approx 15 degrees.
        # Critical (4.5) should be approx 45 degrees.
        TILT_CONST = 5.0 
        self.tilt_angle = self.void_distance * TILT_CONST

    def spin(self, pain_level: float, pleasure_level: float, fog_level: float):
        """
        Apply Bio-Signals to Rotor Physics.
        Intensity (RPM) is driven by the STRONGER emotion (Pain or Pleasure).
        """
        # 1. Determine Intensity & Polarity
        # If Flow/Joy is higher than Pain, we spin positively.
        if pleasure_level >= pain_level:
            intensity = pleasure_level
            self.polarity = 1.0 * intensity # Positive Polarity
        else:
            intensity = pain_level
            self.polarity = -1.0 * intensity # Negative Polarity

        # 2. RPM Calculation
        # Base: 1000
        # Intensity adds up to 9000 RPM (Total 10000)
        self.rpm = 1000 + (intensity * 9000)
        
        # 3. Stability
        # Fog reduces stability
        self.stability = max(0.0, 1.0 - fog_level)

    def check_integrity(self) -> Dict[str, Any]:
        """
        Checks if the Rotor is stable or if it collapses.
        """
        status = "STABLE"
        collapse_reason = None
        
        # Law 1: The Wobble Limit (Tilt)
        # If Tilt > 45 degrees, the Rotor is dangerously divergent.
        if self.tilt_angle > 45.0:
            status = "CRITICAL_TILT"
        
        # Law 2: The Speed Limit (RPM)
        # Extreme RPM can stabilize a tilt (Gyroscopic Effect),
        # but if stability (Fog) is low, it spins out.
        if self.rpm > 8000 and self.stability < 0.3:
            status = "SPIN_OUT"
            collapse_reason = "High RPM with Low Stability (Fever Dream)"

        # Law 3: The Event Horizon (Tension)
        # If Distance > 6.0, the Tether snaps.
        if self.void_distance > 6.0:
            status = "COLLAPSE"
            collapse_reason = "Void Tether SNAPBACK (Distance Critical)"
            
        return {
            "status": status,
            "rpm": self.rpm,
            "tilt": self.tilt_angle,
            "reason": collapse_reason
        }

    def __str__(self):
        return f"<DreamRotor | {self.intent} | RPM:{self.rpm:.0f} | Tilt:{self.tilt_angle:.1f}°>"
