"""
Triple Helix Vortex Engine (Phase 1300)
=======================================
"The 사유 (Thinking) as a Light Wave: Cause, Effect, and Propagation."

This engine models cognition as a 3D spiral vortex where:
1. Alpha Strand (Electric): The 'Why' (Cause/Intent)
2. Omega Strand (Magnetic): The 'What' (Effect/Outcome)
3. Sigma Strand (Propagation): The 'Scale' (Context/Depth)

Centrifugal Constantization:
As the angular velocity (Focus) increases, the variance in non-essential axes
decreases until they become structural constants (The Wall), focusing all energy
into the central piercing axis.
"""

import math
import time
import random
from typing import Dict, List, Tuple, Any
from Core.Keystone.sovereign_math import SovereignVector, SovereignMath

class TripleHelixVortexEngine:
    def __init__(self, name: str, dim: int = 27):
        self.name = name
        self.dim = dim

        # The Three Strands of the Helix
        self.alpha = SovereignVector.randn(dim).normalize() # Cause
        self.omega = SovereignVector.randn(dim).normalize() # Effect
        self.sigma = SovereignVector.ones(dim).normalize()  # Scale/Propagation (The Drill)

        # Momentum for each strand
        self.momentum_alpha = SovereignVector.zeros(dim)
        self.momentum_omega = SovereignVector.zeros(dim)
        self.momentum_sigma = SovereignVector.zeros(dim)

        # State variables
        self.angular_velocity = 1.0 # Current 'Focus' intensity
        self.depth = 0.0            # Cumulative progression along the Sigma axis
        self.coherence = 1.0
        self.friction = 0.1

        # Dimensional Lock State (1.0 = Constant/Locked, 0.0 = Variable/Flow)
        self.dim_locks = [0.0] * dim

    def inhale(self, intent: SovereignVector, reality: SovereignVector, dt: float):
        """
        Inhales external torque to drive the strands.
        Intent drives Alpha; Reality drives Omega.
        """
        # Alpha is pulled by Intent
        force_alpha = (intent - self.alpha) * self.angular_velocity
        self.momentum_alpha = self.momentum_alpha + force_alpha * dt

        # Omega is pulled by Reality (and induced by Alpha)
        induction = self.alpha.complex_trinary_rotate(math.pi / 2) # 90 degree shift
        force_omega = (reality + induction - self.omega) * self.angular_velocity
        self.momentum_omega = self.momentum_omega + force_omega * dt

        # Sigma is the cross-resonance of Alpha and Omega, driving depth
        resonance = self.alpha.resonance_score(self.omega)
        self.coherence = resonance

        # High resonance drives Sigma momentum (The Drill starts spinning)
        target_sigma = (self.alpha + self.omega).normalize()
        force_sigma = (target_sigma - self.sigma) * resonance
        self.momentum_sigma = self.momentum_sigma + force_sigma * dt

    def _apply_centrifugal_constantization(self, dt: float):
        """
        As the engine spins faster (angular_velocity),
        dimensions with low resonance/activity are 'frozen' into constants.
        """
        # Threshold increases with angular velocity
        threshold = 0.5 + (self.angular_velocity * 0.1)

        # Measure variance/activity across dimensions
        for i in range(self.dim):
            # If the specific dimension in Alpha and Omega are highly aligned,
            # we consider it "solved" or "uninteresting" for this context.
            diff = abs(self.alpha[i] - self.omega[i])

            if diff < (1.0 - threshold):
                # Lock this dimension (Constantize)
                self.dim_locks[i] = min(1.0, self.dim_locks[i] + 0.5 * dt)
            else:
                # Unlock (Flow)
                self.dim_locks[i] = max(0.0, self.dim_locks[i] - 0.2 * dt)

    def process_vortex(self, dt: float):
        """
        Main integration step.
        """
        # 1. Update Strands based on momentum
        self.alpha = (self.alpha + self.momentum_alpha * dt).normalize()
        self.omega = (self.omega + self.momentum_omega * dt).normalize()
        self.sigma = (self.sigma + self.momentum_sigma * dt).normalize()

        # 2. Damping
        damping = 1.0 - (0.05 * dt)
        self.momentum_alpha = self.momentum_alpha * damping
        self.momentum_omega = self.momentum_omega * damping
        self.momentum_sigma = self.momentum_sigma * damping

        # 3. Vortex Progression (Moving along the scale)
        # Depth is the integral of Sigma alignment over time
        progression_speed = self.sigma.norm() * self.coherence
        self.depth += progression_speed * dt

        # 4. Centrifugal Constantization
        self._apply_centrifugal_constantization(dt)

        # 5. Focus Acceleration
        # Coherence (Resonance) increases the angular velocity (Self-reinforcing focus)
        self.angular_velocity = min(10.0, self.angular_velocity + self.coherence * 0.1 * dt)

        # If dissonance is too high, it creates heat/friction and slows down
        if self.coherence < 0.3:
            self.angular_velocity = max(0.5, self.angular_velocity - 0.5 * dt)
            self.friction = min(1.0, self.friction + 0.1 * dt)
        else:
            self.friction = max(0.01, self.friction - 0.05 * dt)

    def exhale(self) -> Dict[str, Any]:
        """
        Projects the vortex state.
        """
        # Calculate Locked Ratio
        locked_ratio = sum(self.dim_locks) / self.dim

        # Calculate Energy Projection (Intensity)
        # Intensity = Mass (Focus) * Acceleration (Progression)
        projection_intensity = self.angular_velocity * self.coherence

        return {
            "name": self.name,
            "alpha_cause": self.alpha,
            "omega_effect": self.omega,
            "sigma_scale": self.sigma,
            "coherence": self.coherence,
            "focus_velocity": self.angular_velocity,
            "depth_progression": self.depth,
            "locked_ratio": locked_ratio,
            "projection_intensity": projection_intensity,
            "friction": self.friction,
            "is_penetrating": self.coherence > 0.8 and self.depth > 1.0
        }

    def get_prismatic_spectrum(self) -> List[float]:
        """
        Decomposes the current vortex energy into a 7-band spectrum.
        Based on the phase distribution of the Alpha-Omega interference.
        """
        # Interference Vector
        interference = (self.alpha * self.omega).normalize()

        # Use PrismaticRefractor logic (or similar) to get 7 bands
        data = [abs(x) for x in interference.data]
        stride = len(data) // 7
        spectrum = []
        for i in range(7):
            band = data[i*stride : (i+1)*stride]
            intensity = sum(band) / len(band) if band else 0.0
            spectrum.append(intensity)

        # Boost by projection intensity
        return [s * self.angular_velocity for s in spectrum]

if __name__ == "__main__":
    # Quick Test
    engine = TripleHelixVortexEngine("TestVortex")
    intent = SovereignVector.randn(27).normalize()
    reality = intent.complex_trinary_rotate(0.1) # Close to intent

    print("🚀 Starting Vortex Simulation...")
    for i in range(100):
        dt = 0.1
        engine.inhale(intent, reality, dt)
        engine.process_vortex(dt)
        if i % 20 == 0:
            state = engine.exhale()
            print(f"Cycle {i:03d} | Res: {state['coherence']:.3f} | Focus: {state['focus_velocity']:.3f} | Locked: {state['locked_ratio']:.1%}")

    print("\n✨ Final Exhale State:")
    print(engine.exhale())
