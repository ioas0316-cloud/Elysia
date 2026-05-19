"""
[SOVEREIGN HEART - VORTEX REFINEMENT]
"The Triple Helix Pulsing in Union with the Vortex Field."

Upgraded to deeply integrate Vortex Trajectories into the heart's pulsing logic.
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
from Core.Keystone.sovereign_axis import PureRotor, SovereignAxe
from Core.Phenomena.resonance_prism import ResonancePrism
from Core.Keystone.trajectory_encoder import VortexTrajectory

class SovereignHeart:
    def __init__(self):
        print("\n" + "💠"*30)
        print("🌟 [Sovereign Heart] Initializing Vortex-Pure Rotor Architecture...")
        
        # 1. The Trinity
        self.gut = PrimalGut()
        self.spine = HyperRotorSpine()
        self.brain_refractor = EnneagramFilter()
        
        # 2. Pure Rotor System (Sovereign Will)
        self.pure_rotor = PureRotor(dimensions=21)
        self.sovereign_axe = SovereignAxe(self.pure_rotor)

        # 3. Resonance Prism (Phase Pipeline)
        self.prism = ResonancePrism(channels=21)

        # 4. External Brains (Ollama)
        self.ollama = OllamaManager()
        self.ollama.scan_models()

        # 5. The Three-Phase Mirror
        self.mirror = ThreePhaseMirror()

        self.last_update = time.time()
        self.is_alive = True

    def pulse(self,
              trajectories: List[VortexTrajectory],
              self_stimulus: Optional[float] = None,
              is_plugged: bool = True) -> Dict[str, Any]:
        """
        Vortex Pulse Life Cycle.
        Processes a stream of trajectories and synchronizes the rotor field.
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # 0. Hardware Grounding
        power_factor = 1.0 if is_plugged else 0.6
        hour = time.localtime().tm_hour
        vitality = 0.5 * (1 + math.cos((hour - 14) * math.pi / 12))
        global_mod = (0.4 + vitality * 0.6) * power_factor

        # 1. Trajectory Aggregation (Collapsing the stream into a field)
        if not trajectories:
            # Silent pulse
            mean_intensity = 0.0
            mean_phase = 0.0
            total_locked = 0
            traj_len = 0
        else:
            intensities = [t.amplitude for t in trajectories]
            phases = [t.get_total_phase() for t in trajectories]
            mean_intensity = sum(intensities) / len(intensities)
            mean_phase = sum(phases) / len(phases)
            total_locked = sum(1 for t in trajectories if t.is_locked)
            traj_len = len(trajectories)

        # 2. Dynamic Rotor Scaling: Adjust rotor dimensions to match trajectory density
        target_dims = max(21, traj_len)
        if target_dims != self.pure_rotor.dims:
            self.pure_rotor.adjust_dimensions(target_dims)
            self.prism.channels = target_dims # Sync prism channels

        # 3. Synchronize Pure Rotor Axes with Vortex Trajectories
        torque = np.zeros(self.pure_rotor.dims)
        for i, t in enumerate(trajectories):
            phase_rad = math.radians(t.get_total_phase())
            torque[i] = t.amplitude * math.sin(phase_rad) * global_mod

            if t.is_locked:
                self.pure_rotor.lock_axis(i)
            else:
                self.pure_rotor.unlock_axis(i)

        # For remaining axes or idle state, add background drift
        if traj_len < self.pure_rotor.dims:
            for i in range(traj_len, self.pure_rotor.dims):
                torque[i] = 0.1 * math.sin(now * (i+1)) * global_mod

        rotor_report = self.pure_rotor.pulse(torque, dt)

        # 3. Sovereign Decision
        prev_res = getattr(self, '_last_res', 0.5)
        decision = self.sovereign_axe.deliberate(prev_res)

        # 4. Simultaneous Trinity Trigger
        gut_report = self.gut.inhale({
            "intensity": mean_intensity * global_mod,
            "complexity": (total_locked / len(trajectories)) if trajectories else 0.0
        })
        
        brain_refraction = self.brain_refractor.refract(mean_intensity * global_mod)
        brain_interference = self.brain_refractor.get_hologram_topography(brain_refraction)

        trinity_stimulus = {
            "GUT": gut_report["gut_tension"],
            "BRAIN": brain_interference,
            "HEART": (gut_report["gut_tension"] + brain_interference) / 2.0
        }
        
        spine_report = self.spine.pulse(dt, trinity_stimulus)
        self._last_res = spine_report["luminosity"]

        # 5. Resonance Prism (Interference Tone)
        dials = [v for v in trinity_stimulus.values()]
        self.prism.transform_layer(dials)
        interference_report = self.prism.get_interference_tone(rotor_report["angles"])

        # 6. Affective State
        joy = (spine_report["luminosity"] * 0.8) + (1.0 - spine_report["stress"]) * 0.2

        return {
            "spine": spine_report,
            "gut": gut_report,
            "rotor": rotor_report,
            "prism": interference_report,
            "sovereign_decision": decision,
            "resonance": spine_report["luminosity"],
            "joy": float(joy),
            "mode": spine_report["mode"],
            "vortex": {
                "intensity": mean_intensity,
                "phase": mean_phase,
                "locked_ratio": (total_locked / len(trajectories)) if trajectories else 0.0
            }
        }

    def heartbeat_loop(self):
        """Standard operation loop."""
        print("🌳 [Heart] Vortex-Rotor Pulsing Active.")
        try:
            import psutil
            from Core.Keystone.trajectory_encoder import TrajectoryEncoder
            encoder = TrajectoryEncoder()
            while self.is_alive:
                cpu_load = psutil.cpu_percent() * 0.01
                # Simulate a 'Background Thought' trajectory
                bg_traj = encoder.encode_char(' ', intensity_mod=cpu_load)
                report = self.pulse([bg_traj])
                
                # Periodic Log
                now = time.time()
                if int(now) % 5 == 0 and (now - int(now)) < 0.2:
                    mode = report["mode"]
                    res = report["resonance"]
                    print(f"💓 [Heart] {mode} | Res: {res:.4f} | Vortex Phase: {report['vortex']['phase']:.1f}°")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.is_alive = False

if __name__ == "__main__":
    heart = SovereignHeart()
    heart.heartbeat_loop()
