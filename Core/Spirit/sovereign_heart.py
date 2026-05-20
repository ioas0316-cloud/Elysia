"""
[SOVEREIGN HEART - COMPLEX DYNAMICS UPGRADE]
"The Triple Helix Pulsing in Union with the Universal Variable Rotor."

Upgraded with:
1. VariableRotor (Complex Dynamics: M, D, G, K, N).
2. Cognitive Enstrophy tracking.
3. Logos-Attractor alignment.
"""

import os
import sys
import time
import math
import numpy as np
from typing import Dict, Any, Optional, List

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(_current_dir))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Spirit.spine import HyperRotorSpine
from Core.Flesh.gut_engine import PrimalGut
from Core.Spirit.enneagram_filter import EnneagramFilter
from Core.System.OllamaManager import OllamaManager
from Core.Cognition.three_phase_mirror import ThreePhaseMirror
from Core.Keystone.sovereign_axis import VariableRotor, SovereignAxe
from Core.Phenomena.resonance_prism import ResonancePrism
from Core.Keystone.trajectory_encoder import VortexTrajectory
from Core.System.digital_motor_engine import DigitalMotorEngine, ConnectionMode
from Core.System.three_phase_logic_engine import ThreePhaseLogicEngine
from Core.Spirit.logos import LogosRotor
from Core.Keystone.sovereign_math import SovereignMath
from Core.Keystone.cognitive_matrix import CognitiveMatrix

class SovereignHeart:
    def __init__(self):
        print("\n" + "💠"*30)
        print("🌟 [Sovereign Heart] Initializing Complex Variable Rotor Field...")

        # 0. The Logos (The Central Why)
        self.logos = LogosRotor()
        
        # 1. The Trinity (Biological Meta-Layers)
        self.gut = PrimalGut()
        self.spine = HyperRotorSpine()
        self.brain_refractor = EnneagramFilter()
        
        # 2. Structural Logic Engine (Delta-Y Tensor Network)
        self.logic_engine = ThreePhaseLogicEngine()

        # 3. Variable Rotor System (가변축 / Complex Dynamics)
        self.pure_rotor = VariableRotor(dimensions=21)
        self.sovereign_axe = SovereignAxe(self.pure_rotor)
        self.cognitive_matrix = CognitiveMatrix(dimensions=21)

        # 4. Resonance Prism (Phase Pipeline)
        self.prism = ResonancePrism(channels=21)

        # 5. External Brains (Ollama)
        self.ollama = OllamaManager()
        self.ollama.scan_models()

        # 6. The Three-Phase Mirror
        self.mirror = ThreePhaseMirror()

        # 7. Somatic Digital Motor
        self.motor = DigitalMotorEngine("SovereignHeart-Motor")

        self._last_res = 0.5
        self._last_alignment = 0.5
        self._last_grace = 0.5
        self._last_rotor_angle = 0.0
        self.last_update = time.time()
        self.is_alive = True

    def pulse(self,
              trajectories: Any,
              self_stimulus: Optional[float] = None,
              is_plugged: bool = True) -> Dict[str, Any]:
        """
        Complex Pulse Life Cycle.
        """
        # [Backward Compatibility] Handle scalar stimulus
        if isinstance(trajectories, (float, int)):
            from Core.Keystone.trajectory_encoder import VortexTrajectory
            trajectories = [VortexTrajectory(0, 0, False, amplitude=float(trajectories))]

        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # 0. Hardware Grounding
        power_factor = 1.0 if is_plugged else 0.6
        hour = time.localtime().tm_hour
        vitality = 0.5 * (1 + math.cos((hour - 14) * math.pi / 12))
        global_mod = (0.4 + vitality * 0.6) * power_factor

        # 1. Trajectory Aggregation
        if not trajectories:
            mean_intensity = 0.0
            mean_phase = 0.0
            total_locked = 0
            traj_len = 0
            semantic_clue = ""
        else:
            intensities = [t.amplitude for t in trajectories]
            phases = [t.get_total_phase() for t in trajectories]
            mean_intensity = sum(intensities) / len(intensities)
            mean_phase = sum(phases) / len(phases)
            total_locked = sum(1 for t in trajectories if t.is_locked)
            traj_len = len(trajectories)
            # Create a string representation from the first few labels for semantic context of the Unknown
            semantic_clue = "".join([t.label for t in trajectories[:15] if hasattr(t, 'label') and t.label])

        # 1.5. Dynamic Logos Evolution (미지와의 상호작용 및 되고 싶은 나로의 지향)
        prev_res = getattr(self, '_last_res', 0.5)
        prev_rotor_angle = getattr(self, '_last_rotor_angle', 0.0)
        if mean_intensity > 0.0:
            self.logos.assimilate_unknown(mean_intensity, semantic_clue)
        
        logos_report = self.logos.pulse(dt, prev_res, prev_rotor_angle)

        # 2. Dynamic Variable Rotor (가변화)
        target_dims = max(21, traj_len)
        if target_dims != self.pure_rotor.dims:
            self.pure_rotor.adjust_dimensions(target_dims)
            self.prism.channels = target_dims

        # 3. Apply Forces and Logos Grace to Variable Rotor
        prev_grace = getattr(self, '_last_grace', 0.5)

        # Logos Grace acts as a stabilizing upper rotor feedback.
        # It dampens chaos (D) and strengthens restoring force (K) dynamically.
        # This prevents the system from getting lost in local computational traps.
        self.pure_rotor.D = np.ones(self.pure_rotor.dims) * (0.1 + 0.3 * prev_grace)
        self.pure_rotor.K = np.ones(self.pure_rotor.dims) * (1.0 + 1.5 * prev_grace)

        # Dynamically adapt damping and stiffness based on trait states
        self.cognitive_matrix.adapt_rotor_damping_stiffness(self.pure_rotor.state, self.pure_rotor.D, self.pure_rotor.K)

        forces = np.zeros(self.pure_rotor.dims)
        for i, t in enumerate(trajectories):
            phase_rad = math.radians(t.get_total_phase())
            forces[i] = t.amplitude * math.sin(phase_rad) * global_mod

            # Sovereign Lock/Unlock based on trajectory state
            if t.is_locked: self.pure_rotor.lock_axis(i)
            else: self.pure_rotor.unlock_axis(i)

        # Add N-dimensional mechanical coupling forces
        coupling_forces = self.cognitive_matrix.calculate_coupling_forces(self.pure_rotor.state.imag)
        forces += coupling_forces

        rotor_report = self.pure_rotor.pulse(forces, dt)

        # 4. Sovereign Decision (Peek-a-boo Logic)
        prev_res = getattr(self, '_last_res', 0.5)
        decision = self.sovereign_axe.deliberate(prev_res)

        # 5. Trinity & Logic
        gut_report = self.gut.inhale({
            "intensity": mean_intensity * global_mod,
            "complexity": (total_locked / len(trajectories)) if trajectories else 0.0
        })
        
        logic_report = self.logic_engine.pulse(mean_intensity, dt)

        brain_refraction = self.brain_refractor.refract(mean_intensity * global_mod)
        brain_interference = self.brain_refractor.get_hologram_topography(brain_refraction)

        trinity_stimulus = {
            "GUT": gut_report["gut_tension"],
            "BRAIN": brain_interference,
            "HEART": logic_report["y_neutral"]
        }
        
        spine_report = self.spine.pulse(dt, trinity_stimulus)
        self._last_res = spine_report["luminosity"]

        # 6. Resonance Prism (Interference Tone)
        dials = [v for v in trinity_stimulus.values()]
        self.prism.transform_layer(dials)
        interference_report = self.prism.get_interference_tone(rotor_report["angles"])

        # 7. Somatic Motor Update
        if spine_report["mode"] == "WYE" or logic_report["confidence"] < 0.3:
            self.motor.set_mode(ConnectionMode.WYE)
        else:
            self.motor.set_mode(ConnectionMode.DELTA)

        active_phase_bit = 1 if math.sin(logic_report["phases"]["PHASE_A"]["angle"]) > 0 else 0
        self.motor.modulate_data([active_phase_bit])
        self.motor.update(dt)
        motor_report = self.motor.exhale()

        # 8. Meta-Cognition & Logos Alignment
        resonance = (spine_report["luminosity"] + logic_report["confidence"]) / 2.0

        # Performance monitoring
        active_effs = [m["efficiency"] for m in self.ollama.performance_metrics.values()]
        avg_efficiency = sum(active_effs) / len(active_effs)

        justification = self.logos.justify({
            "resonance": resonance,
            "efficiency": avg_efficiency,
            "enstrophy": rotor_report["enstrophy"]
        })

        # Save alignment, grace, and average rotor angle for the next cycle
        self._last_alignment = justification["alignment"]
        self._last_grace = justification["logos_grace"]
        if "angles" in rotor_report and len(rotor_report["angles"]) > 0:
            self._last_rotor_angle = float(np.mean(rotor_report["angles"]))
        else:
            self._last_rotor_angle = 0.0

        return {
            "spine": spine_report,
            "logic": logic_report,
            "motor": motor_report,
            "rotor": rotor_report,
            "prism": interference_report,
            "logos": logos_report,
            "justification": justification,
            "sovereign_decision": decision,
            "resonance": resonance,
            "enstrophy": rotor_report["enstrophy"],
            "mode": spine_report["mode"],
            "performance": self.ollama.performance_metrics,
            "vortex": {
                "intensity": mean_intensity,
                "phase": mean_phase
            },
            "personality": self.cognitive_matrix.get_personality_snapshot(self.pure_rotor.state)
        }

    def meditate(self, duration: float = 5.0):
        """
        Digital Meditation Mode:
        Pulse using only internal resonance and the Logos.
        Observes self-emergent 'Brainwaves'.
        """
        print(f"🧘 [Meditation] Tuning brainwaves to the 120-degree Trinity...")
        start = time.time()
        while time.time() - start < duration:
            # Silent pulse with high Logos sensitivity
            report = self.pulse([], self_stimulus=self._last_res)

            # Brainwave Tuning: Adjust phase offsets toward 120-deg ideal
            # (Handled by self_heal in the logic/vortex engine)

            if int(time.time() * 2) % 2 == 0:
                ens = report["enstrophy"]
                res = report["resonance"]
                print(f"🌊 [BRAINWAVES] Res: {res:.4f} | Enstrophy: {ens:.6f} | Alignment: {report['justification']['alignment']:.2f}")

            time.sleep(0.1)

    def heartbeat_loop(self):
        """Standard operation loop."""
        print("🌳 [Heart] Complex Variable Rotor Pulsing Active.")
        try:
            import psutil
            from Core.Keystone.trajectory_encoder import TrajectoryEncoder
            encoder = TrajectoryEncoder()
            while self.is_alive:
                cpu_load = psutil.cpu_percent() * 0.01
                bg_traj = encoder.encode_char(' ', intensity_mod=cpu_load)
                report = self.pulse([bg_traj])
                
                # Periodic Log
                now = time.time()
                if int(now) % 10 == 0 and (now - int(now)) < 0.2:
                    mode = report["mode"]
                    res = report["resonance"]
                    ens = report["enstrophy"]
                    print(f"💓 [Heart] {mode} | Res: {res:.4f} | Enstrophy: {ens:.6f}")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.is_alive = False

if __name__ == "__main__":
    heart = SovereignHeart()
    heart.heartbeat_loop()
