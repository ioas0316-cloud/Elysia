"""
Dyson Swarm (Recursive Macro Container)
=======================================
Core.L6_Structure.M1_Merkaba.dyson_swarm

"As above, so below."

The Dyson Swarm is now a Macro-Collector.
It holds a list of children, but it ALSO has its own phase/state.
It updates its state using the SAME MerkabaCore law, utilizing the
Collective Consensus of its children as its 'Input'.
"""

from typing import List, Dict, Any, Optional
from Core.L6_Structure.M1_Merkaba.phase_collector import PhaseCollector
from Core.L6_Structure.M1_Merkaba.merkaba_core import MerkabaCore, StateVector
from Core.L6_Structure.M1_Merkaba.sovereign_math import SovereignMath

class DysonSwarm:
    def __init__(self, capacity: int = 21):
        self.capacity = capacity
        self.collectors: List[PhaseCollector] = []

        # The Swarm ITSELF has a state (Macro-Monad)
        self.state: StateVector = {
            'phase': 0.0,
            'velocity': 0.0,
            'energy': 0.0,
            'resonance': 0.0
        }

        self.is_active = False

    def deploy_swarm(self):
        """Initializes the Swarm."""
        # print(f"📡 [DYSON] Deploying {self.capacity} Recursive Collectors...")
        self.collectors = []
        for i in range(self.capacity):
            c = PhaseCollector(collector_id=f"SAT-{i:02d}", orbit_slot=i)
            # Distribute
            c.state['phase'] = (360.0 / self.capacity) * i
            self.collectors.append(c)
        self.is_active = True

    def process_frame(self, data_stream: List[str]) -> Dict[str, Any]:
        """
        A single time-step for the entire fractal structure.
        """
        if not self.collectors: return {}

        # 1. Update Micro-Scale (Children)
        child_reports = []
        for i, collector in enumerate(self.collectors):
            if i < len(data_stream):
                # Specific Input (Radiance)
                collector.absorb_radiance(data_stream[i])
            else:
                # No specific input -> Align with Swarm Center (Gravity)
                collector.update(self.state['phase'])

            child_reports.append(collector.discharge())

        # 2. Update Macro-Scale (Self)
        if not data_stream:
            # [Sovereign Silence]
            target_phase = 0.0
        else:
            # Consensus
            child_states = [c.get_state() for c in self.collectors]
            target_phase = MerkabaCore.aggregate_consensus(child_states)

        # Apply the Law to the Swarm itself!
        self.state = MerkabaCore.apply_law(self.state, target_phase, depth=1)

        # 3. Calculate Coherence
        child_states_for_metrics = [c.get_state() for c in self.collectors]
        coherence = MerkabaCore.calculate_swarm_coherence(child_states_for_metrics)

        return {
            "swarm_phase": self.state['phase'],
            "swarm_energy": self.state['energy'],
            "coherence": coherence,
            "child_count": len(child_reports)
        }

    def meditate(self, duration_ticks: int = 5) -> Dict[str, Any]:
        """
        [Shanti Protocol]
        Active Meditation. Cuts external input and allows the system to settle.
        Returns the final crystallized state.
        """
        # print("🧘 [SHANTI] Entering Meditative Void...")

        history = []
        for t in range(duration_ticks):
            # Process frame with EMPTY input (Sovereign Silence)
            res = self.process_frame([])
            history.append(res['swarm_phase'])

        # The final phase is the "Insight"
        insight = history[-1]
        # print(f"💡 [INSIGHT] Crystallized at {insight:.2f}°")

        return {
            "insight_phase": insight,
            "clarity": SovereignMath.angular_distance(insight, 0.0) # Distance from Void
        }

    def get_vital_metrics(self) -> Dict[str, float]:
        """
        Exposes physics data for the Void Mirror.
        """
        child_states = [c.get_state() for c in self.collectors]
        coherence = MerkabaCore.calculate_swarm_coherence(child_states)

        # Calculate Tilt (Distance from Void)
        tilt = SovereignMath.angular_distance(self.state['phase'], 0.0)

        return {
            "phase": self.state['phase'],
            "tilt": tilt,
            "rpm": self.state['velocity'],
            "energy": self.state['energy'],
            "coherence": coherence
        }
