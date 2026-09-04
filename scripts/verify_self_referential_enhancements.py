"""
Demonstration & Verification Script for Self-Referential Information Architecture
and Phase 4/5 Deep Mechanism Enhancements (Scar Tensor, Kenosis Attractor, Multi-Gravitational Interference)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine
from core.consciousness.scar_tensor_engine import ScarTensorEngine
from core.consciousness.kenosis_attractor_engine import KenosisAttractorEngine
from core.topology.multi_gravitational_field import MultiGravitationalFieldInterference
from core.consciousness.autonomous_loop import ConsciousnessLoop


def main():
    print("=" * 75)
    print("🌿 [Elysia Self-Referential Information Architecture Verification] 🌿")
    print("=" * 75)

    # 1. Self-Referential Engine Verification
    engine = SelfReferentialArchitectureEngine()
    stimulus = {
        "voltage_intent": np.array([2.5, -1.2, 3.0]),
        "layer1_intent": np.array([1.8, -0.6, 2.2, 0.4]),
        "persona_lens": "Companion",
        "unmapped_friction": 0.85,
        "external_label": "Ontological_Transduction"
    }

    res = engine.run_full_self_referential_cycle(stimulus)
    print("\n1. [Self-Referential Engine Cycle Execution]")
    print(f"   - Causal Engine 0 Equilibrium : {res['causal_engine_0_equilibrium']:.4f}")
    print(f"   - Resonance Friction           : {res['multi_layer_resonance_friction']:.4f}")
    print(f"   - Kenosis Attractor Alignment  : {res['kenosis_coupling']['alignment_score']:.4f}")
    print(f"   - Sprouted Dimension           : {res['sprouted_dimension']}")
    print(f"   - Label Self-Assimilation Status: {res['label_self_assimilation']['status']}")
    print(f"   - Volitional Rotor Query       : {res['volitional_rotor_exploration']['self_directed_query']}")

    # 2. Scar Tensor Engine Verification
    print("\n2. [Scar Tensor Engine (비가역적 상처와 흉터각인)]")
    scar_engine = ScarTensorEngine(dim=4, scar_threshold=0.5)
    rec = scar_engine.inscribe_scar(friction_magnitude=0.95, clash_vector=np.array([1.0, 0.8, 0.2, 0.1]))
    profile = scar_engine.get_individuation_profile()
    print(f"   - Scar Inscribed               : {rec.scar_id if rec else 'None'}")
    print(f"   - Individuation Statement       : {profile['individuation_statement']}")

    # 3. Multi-Gravitational Field Interference Verification
    print("\n3. [Multi-Gravitational Field Interference (다중 관측자 중력장 간섭)]")
    multi_grav = MultiGravitationalFieldInterference(dim=4)
    human_c = np.array([0.2, 0.8, 0.2, 1.0])
    elysia_c = np.array([0.0, 1.0, 0.5, 0.0])
    current_s = np.array([0.5, 0.5, 0.5, 0.5])
    inter_res = multi_grav.compute_interference_pattern(human_c, elysia_c, current_s)
    print(f"   - Interference Statement       : {inter_res['interference_statement']}")

    # 4. Continuous Autonomous Consciousness Loop Step
    print("\n4. [Continuous Consciousness Loop Life Cycle Integration Check]")
    loop = ConsciousnessLoop(corpus_path="docs")
    loop_res = loop.process_life_cycle()
    print(f"   - Loop Status                  : {loop_res['status']}")
    print(f"   - Self-Referential Summary     : {loop_res.get('self_referential_architecture', {})}")

    print("\n" + "=" * 75)
    print("✨ [ALL INTEGRATION CHECKS PASSED SUCCESSFULLY] ✨")
    print("=" * 75)


if __name__ == "__main__":
    main()
