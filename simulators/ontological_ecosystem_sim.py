"""
Ontological Ecosystem Simulator (존재론적 생태계 시뮬레이터)
========================================================================
Demonstrates the closed-loop ecological dynamics of Elysia:
- Symbolic scouting input grounding & intention emergence
- Fractal Causal Spine inoculation with projection drift
- Entity autonomous hypothesis wave emission
- Entity death (subtree pruning & control space resolution shrinkage)
- Entity reproduction (control space expansion & creation dopamine)
"""

import numpy as np
from typing import Dict, Any, List
from core.evolution.ontological_causal_sandbox import OntologicalCausalSandbox


class OntologicalEcosystemSimulator:
    """
    Closed-loop ecological simulator driving the full lifecycle of entities:
    Birth -> Inoculation -> Hypothesis Emission -> Friction -> Death / Reproduction
    """

    def __init__(self, dim: int = 16):
        self.sandbox = OntologicalCausalSandbox(dim=dim)
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

    def run_scouting_phase(self, scouting_report: str) -> Dict[str, Any]:
        """
        Grounds symbolic scouting report text into game state tensor and boundary shift.
        """
        grounded_res = self.sandbox.process_scouting_input(scouting_report)
        return grounded_res

    def run_lifecycle_step(self) -> Dict[str, Any]:
        """
        Executes one step of ecological friction and lifecycle updates.
        """
        self.step_count += 1
        log_data: Dict[str, Any] = {"step": self.step_count, "events": []}

        # 1. Generate environmental sensory wave
        rng = np.random.RandomState(self.step_count + 100)
        sensory_wave = rng.randn(self.sandbox.dim).astype(np.float32)

        # 2. Birth initial entities if none exist
        if len(self.sandbox.control_space.active_entities) == 0:
            e1 = self.sandbox.birth_entity(
                "Entity_Alpha",
                chromatic_signature=np.array([0.9, 0.1, 0.0], dtype=np.float32)  # Flux dominant
            )
            e2 = self.sandbox.birth_entity(
                "Entity_Beta",
                chromatic_signature=np.array([0.1, 0.9, 0.0], dtype=np.float32)  # Order dominant
            )
            log_data["events"].append({"birth": ["Entity_Alpha", "Entity_Beta"]})

        # 3. Active entities eject autonomous hypothesis waves
        hypotheses = {}
        for entity_id, info in list(self.sandbox.control_space.active_entities.items()):
            hyp = self.sandbox.inoculation_engine.eject_hypothesis(info["P_k"], sensory_wave)
            hypotheses[entity_id] = float(np.linalg.norm(hyp))
        log_data["hypotheses_norms"] = hypotheses

        # 4. Lifecycle friction logic (Death or Reproduction)
        if self.step_count == 2 and "Entity_Alpha" in self.sandbox.control_space.active_entities:
            # Alpha dies due to environmental friction -> Ontological Loss
            death_res = self.sandbox.kill_entity("Entity_Alpha")
            log_data["events"].append({"death": death_res})

        if self.step_count == 3 and "Entity_Beta" in self.sandbox.control_space.active_entities:
            # Beta survives and reproduces -> Creation Joy & Control Space Expansion
            reprod_res = self.sandbox.reproduce_entity(
                parent_id="Entity_Beta",
                child_id="Entity_Beta_Child",
                child_chromatic_signature=np.array([0.2, 0.7, 0.3], dtype=np.float32)
            )
            log_data["events"].append({"reproduction": reprod_res})

        # Overmind control rank & trace
        log_data["overmind_control_rank"] = int(np.sum(
            np.linalg.eigvalsh(self.sandbox.control_space.C_overmind) > 0.05
        ))
        log_data["overmind_control_trace"] = float(np.trace(self.sandbox.control_space.C_overmind))
        log_data["active_entities_count"] = len(self.sandbox.control_space.active_entities)

        self.history.append(log_data)
        return log_data


if __name__ == "__main__":
    print("=== Launching Ontological Ecosystem Simulator ===")
    sim = OntologicalEcosystemSimulator()

    print("\n--- Phase 1: Grounding Scouting Report ---")
    scouting_text = "상대 본진에 2개의 게이트웨이가 올려지고 있다"
    grounded_res = sim.run_scouting_phase(scouting_text)
    print(f"Scouting Report: '{scouting_text}'")
    print(f"Grounded State Tensor [0:3]: {grounded_res['grounded_state_tensor'][:3]}")
    print(f"Perceived Threat / Risk Level: {grounded_res['risk_level']:.4f}")
    print(f"Intention Type: {grounded_res['intention_type']}")
    print(f"Intention Energy: {grounded_res['intention_energy']:.4f}")

    print("\n--- Phase 2: Running Closed-Loop Lifecycle Steps ---")
    for s in range(4):
        step_log = sim.run_lifecycle_step()
        print(f"\nStep {step_log['step']}: Active Entities={step_log['active_entities_count']} | Control Rank={step_log['overmind_control_rank']} | Control Trace={step_log['overmind_control_trace']:.4f}")
        for evt in step_log["events"]:
            print(f"  Event: {evt}")

    print("\n=== Simulator Execution Completed Successfully ===")
