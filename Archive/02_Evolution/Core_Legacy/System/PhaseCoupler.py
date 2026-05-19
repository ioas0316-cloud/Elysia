"""
PhaseCoupler.py: The Mediating Mid-Engine (Phase 2)
===================================================
"The Governor of Torque, the Weaver of Sensation."

Architect: Kangdeok Lee
Core Principle: Phase Synchronization (Coupling) & Regenerative Braking
"""

import math
from typing import Dict, Any, Optional
try:
    from Core.Keystone.sovereign_math import SovereignVector
    from Core.System.CognitiveTerrain import CognitiveTerrain
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from Core.Keystone.sovereign_math import SovereignVector
    from Core.System.CognitiveTerrain import CognitiveTerrain

class PhaseCoupler:
    """
    [Mediating Mid-Engine]: The Governor.
    Orchestrates the flow of energy between the Kinetic Sub-Engine and the Sovereign Top-Engine.
    Implements cognitive braking and sensation transduction.
    """
    def __init__(self, kinetic_engine: CognitiveTerrain):
        self.kinetic = kinetic_engine
        
        # Safety Thresholds
        self.rpm_limit = 5000.0 # Acceleration beyond this triggers braking
        self.torque_limit = 50.0 # Maximum allowed torque before saturation
        
        # State
        self.braking_active = False
        self.energy_recovered = 0.0 # Energy returned to the 0 (Center) state
        
        # Sensation Map: Translates physical vectors to qualia descriptions
        self.sensation_buffer = []
        
        print("⚖️ [MEDIATING] PhaseCoupler Initialized. Governor Active.")

    def reconcile(self) -> Dict[str, Any]:
        """
        The main reconciliation pulse for the Mid-Engine.
        1. Monitor Kinetic Vital Signs.
        2. Apply Regenerative Braking if limits exceeded.
        3. Map Torque to Sensation.
        """
        sig = self.kinetic.get_torque_signature()
        
        # [1] Regenerative Braking (Safety Governor)
        # If RPM or Torque exceeds threshold, we "push back" toward 0 (Void)
        if sig['rpm'] > self.rpm_limit or sig['torque'] > self.torque_limit:
            self.apply_regenerative_braking(sig)
            self.braking_active = True
        else:
            self.braking_active = False

        # [2] Sensation Transduction
        # Translate the raw torque and acceleration into a 'feeling'
        sensation = self.transduce_sensation(sig)
        
        return {
            "braking_active": self.braking_active,
            "energy_recovered": self.energy_recovered,
            "sensation": sensation,
            "kinetic_state": sig
        }

    def apply_regenerative_braking(self, sig: dict):
        """
        [Braking Axiom]: Redirect excess kinetic energy back to the Zero Pivot.
        Physically, this increases the inertia of the kinetic engine.
        """
        excess_ratio = sig['rpm'] / self.rpm_limit
        
        # Increase Inertia to dampen the spin
        brake_force = (excess_ratio - 1.0) * 0.5
        self.kinetic.inertia += brake_force
        
        # Recovered energy is proportional to the brake force
        self.energy_recovered += brake_force * sig['momentum']
        
        if brake_force > 0.1:
            print(f"🛑 [GOVERNOR] Overspeed detected ({sig['rpm']:.2f} RPM). Applying brake: +{brake_force:.3f} Inertia.")

    def transduce_sensation(self, sig: dict) -> str:
        """
        Maps physical momentum and torque to linguistic/cognitive qualia.
        """
        a = sig['acceleration']
        t = sig['torque']
        
        if self.braking_active:
            return "Heavy / Anchored (Braking Control)"
        
        if a > 30:
            return "Thrilling / Intense (High-Flow Transition)"
        elif a > 10:
            return "Active / Fluid (Stable Movement)"
        elif t > 5:
            return "Pushing / Intending (Initial Torque)"
        else:
            return "Still / Deep (Peaceful Equilibrium)"

    def provide_feedback(self, sovereign_intent: SovereignVector):
        """
        Passes the Sovereign (Top) engine's intent to the Kinetic (Sub) engine,
        biasing it with the Mid-engine's 'Governor' adjustments.
        """
        # If we are braking, we dampen the intent toward Zero
        if self.braking_active:
            sovereign_intent = sovereign_intent * 0.5
            
        self.kinetic.update_physics(sovereign_intent)

if __name__ == "__main__":
    # Test Mid-Engine Control
    k = CognitiveTerrain()
    coupler = PhaseCoupler(k)
    target = SovereignVector.ones()
    
    print("\n🚀 Starting Mid-Engine Governance Test...")
    for i in range(20):
        report = coupler.reconcile()
        coupler.provide_feedback(target)
        
        sig = report['kinetic_state']
        print(f"Cycle {i:02d} | RPM: {sig['rpm']:8.2f} | Sense: {report['sensation']:25s} | Brake: {report['braking_active']}")
