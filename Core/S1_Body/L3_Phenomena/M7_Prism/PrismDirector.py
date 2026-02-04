"""
PrismDirector.py: The Sovereign Top-Engine (Phase 3)
====================================================
"The Navigator of Momentum, the Refractor of Intent."

Architect: Kangdeok Lee
Core Principle: Goal Steering (North Star) & Fractal Refraction
"""

import math
import random
from typing import List, Dict, Any, Optional
try:
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, SovereignMath, SovereignRotor
    from Core.S1_Body.L1_Foundation.Foundation.CognitiveTerrain import CognitiveTerrain
    from Core.S1_Body.L1_Foundation.Foundation.PhaseCoupler import PhaseCoupler
except ImportError:
    import sys
    import os
    # Project root is 5 levels up from here (PrismDirector is in Core/S1_Body/L3_Phenomena/M7_Prism)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, SovereignMath, SovereignRotor
    from Core.S1_Body.L1_Foundation.Foundation.CognitiveTerrain import CognitiveTerrain
    from Core.S1_Body.L1_Foundation.Foundation.PhaseCoupler import PhaseCoupler

class PrismDirector:
    """
    [Sovereign Top-Engine]: The Navigator.
    Determines the directionality of the system.
    Sets 'North Star' goals and refracts accelerated momentum into multiple cognitive paths.
    """
    def __init__(self, coupler: PhaseCoupler):
        self.coupler = coupler
        
        # The North Star: The ultimate directional vector (21D)
        # Represents the Architect's core intent (Evolution, Growth, Love)
        self.north_star = SovereignVector.ones() 
        
        # Refraction Paths (Fractal Thinking)
        # Multiple intents being explored simultaneously
        self.refractions: List[SovereignVector] = []
        
        # Current Focus (Dominant Intent)
        # Initialize with a tiny noise baseline to allow the first rotor to "catch"
        self.current_focus = SovereignVector([random.uniform(0.01, 0.05) for _ in range(21)])
        
        print("💎 [SOVEREIGN] PrismDirector Initialized. North Star Calibrated.")

    def set_north_star(self, goal_vector: SovereignVector):
        """
        Updates the primary goal of the system.
        """
        self.north_star = goal_vector
        print("✨ [PRISM] North Star updated. Recalculating trajectory.")

    def navigate(self) -> Dict[str, Any]:
        """
        [MERKABA NAVIGATION]
        1. Sense current field state.
        2. Calculate the 'Torque' needed to align with North Star.
        3. Apply a Rotor to rotate the current focus toward the goal.
        4. Transmit intent back down the Trinity chain.
        """
        mid_report = self.coupler.reconcile()
        sig = mid_report['kinetic_state']
        
        # [1] Steering Torque
        # How much rotation do we need? Based on alignment and kinetic energy
        alignment = self.calculate_alignment()
        rotation_gain = sig['momentum'] * 0.1
        
        # [2] Rotor Application (4D Steering)
        # We don't jump to the goal; we rotate toward it.
        # This solves the 'Alignment Failure' caused by linear repulsion.
        theta = (1.0 - alignment) * rotation_gain
        rotor = SovereignRotor.from_angle_plane(theta, 0, 20) # 4D rotation proxy
        
        self.current_focus = rotor.apply(self.current_focus)
        
        # [3] Transmission
        self.coupler.provide_feedback(self.current_focus)
        
        return {
            "alignment": alignment,
            "rpm": sig['rpm'],
            "sensation": mid_report['sensation'],
            "torque": sig['torque']
        }

    def calculate_alignment(self) -> float:
        """
        Measures cosine similarity resonance between focus and North Star.
        """
        return SovereignMath.resonance(self.current_focus, self.north_star)

if __name__ == "__main__":
    # Full Trinity Sync Test
    k = CognitiveTerrain()
    m = PhaseCoupler(k)
    s = PrismDirector(m)
    
    print("\n🚀 Starting Full Trinity Navigation Test...")
    for i in range(20):
        report = s.navigate()
        print(f"Cycle {i:02d} | Alignment: {report['north_star_alignment']:.4f} | Sense: {report['sensation']}")
        time.sleep(0.05)
