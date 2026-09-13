r"""
Causal Conservation Node (CC-Node) & Isomorphic Knowledge Assimilation Demonstration
=====================================================================================

Demonstrates:
1. 4-Phase Isomorphic Knowledge Assimilation Loop & Multi-layer Ecosystem Trophic Cascade.
2. CC-Node Tensor Dynamics: 3 Structural Layers (Bound Tension Field, Topological Skeleton, Generative Rule Engine)
   & 3 Operational Mechanics (Unfolding, Tension Resistance, Reversible Implosion).
3. Naive Vector Representation vs. CC-Node Dynamics comparison.
4. Causal Game Mechanics Engine & SealedAttractor Narrative Recovery.
"""

import torch
import numpy as np
from core.engine.assimilation_loop import (
    ExternalCausalSignal,
    IsomorphicKnowledgeAssimilationLoop,
    PhaseTopologicalReconstructionEngineExt,
)
from synaptic_architecture.causal_conservation_node import CausalConservationNode
from synaptic_architecture.causal_game_engine import CausalGameMechanicsEngine


def run_demo():
    print("==========================================================================")
    print("   Causal Conservation Node (CC-Node) Engine Demonstration               ")
    print("==========================================================================")

    # 1. Isomorphic Knowledge Assimilation Loop (Multi-Layer Ecosystem Trophic Cascade)
    print("\n[1/4] Running Isomorphic Knowledge Assimilation Loop (Ecosystem Cascade)...")
    loop = IsomorphicKnowledgeAssimilationLoop(resonance_threshold=0.08)

    signal = ExternalCausalSignal(
        signal_id="ECO_TROPHIC_CASCADE_01",
        boundary_conditions=[
            "ENERGY_CONSERVATION_BETWEEN_TROPHIC_LEVELS",
            "NEGATIVE_FEEDBACK_ATTRACTOR_REQUIRED",
            "DISCONTINUOUS_EXTINCTION_PREVENTED",
        ],
        causal_relationships={
            "Apex_Predator_Population": "Herbivore_Foraging_Behavior",
            "Herbivore_Foraging_Behavior": "Vegetation_Root_Density",
            "Vegetation_Root_Density": "Soil_Erosion_Resistance",
            "Soil_Erosion_Resistance": "Drought_Resilience_Capacity",
            "Drought_Resilience_Capacity": "Apex_Predator_Population",
        },
    )

    # Step 1: Deconstruction
    extracted = loop.deconstruct_and_extract(signal)
    print(f"    - [Step 1 Extracted Invariants]: {len(extracted['core_morphisms'])} core morphisms")

    # Step 2: Isomorphic Mapping
    mapped = loop.isomorphic_map(extracted)
    print(f"    - [Step 2 Isomorphic Graph]: mapped {len(mapped['mapped_graph'])} internal operators")

    # Step 3: Generative Simulation & Friction Tension (V_t)
    tension = loop.run_internal_simulation(mapped)
    print(f"    - [Step 3 Internal Simulation]: Causal Friction Tension V_t = {tension:.4f}")

    # Step 4: Phase Resonance & CC-Node Freeze
    cc_node_res = loop.assimilate_knowledge(signal)
    if cc_node_res:
        print(f"    - [Step 4 Resonance Success]: Frozen CC-Node ID = {cc_node_res.node_id}")

    # 2. PyTorch CC-Node Tensor Mechanics
    print("\n[2/4] Initializing PyTorch CC-Node Engine Dynamics (Dim=32)...")
    dim = 32
    torch_node = CausalConservationNode(node_id="CC_NODE_PHYSICAL_WAVE", dim=dim)

    batch_size = 2
    context = torch.randn(batch_size, dim)

    # Forward evaluation
    v_potential, v_react, i_c, constraints = torch_node(context, c_lens_scale=1.0)
    print(f"    - Layer 1 Potential Energy V_potential Norm: {v_potential.mean().item():.4f}")
    print(f"    - Layer 2 Topological Invariant Spectrum I_c: {i_c.tolist()[:4]}...")
    print(f"    - Layer 3 Generated Boundary Constraints Shape: {constraints.shape}")

    # Mechanic 1: Unfolding
    unfolded_traj = torch_node.unfold(context, steps=3)
    print(f"    - Mechanic 1 (Unfolding): Generated Trajectory Shape = {unfolded_traj.shape}")

    # Mechanic 2: Tension Resistance under contradiction noise
    extreme_noise = torch.randn(batch_size, dim) * 5.0
    v_react_noise, norm_tension = torch_node.compute_tension_resistance(extreme_noise)
    print(f"    - Mechanic 2 (Tension Resistance): Reaction Tension Norm = {norm_tension:.4f}")

    # Mechanic 3: Reversible Implosion (SealedAttractor)
    original_state = torch_node.latent_state.clone()
    vault = torch_node.implode_and_seal()
    print(f"    - Mechanic 3 (Implosion): Node sealed into attractor vault (Sealed={torch_node.is_sealed})")

    torch_node.latent_state.data.add_(100.0)  # Perturb active memory
    torch_node.unseal_and_recover()
    is_lossless = torch.allclose(torch_node.latent_state, original_state)
    print(f"    - Mechanic 3 (Unsealing): Lossless State Recovery Confirmed = {is_lossless}")

    # 3. Naive Vector Representation vs. CC-Node Comparison
    print("\n[3/4] Naive Numeric Vector vs CC-Node Mechanics Comparison:")
    print("    ┌───────────────────────────┬───────────────────────────────┬────────────────────────────────┐")
    print("    │ Property                  │ Naive Numeric Vector / Token  │ Causal Conservation Node (CC)  │")
    print("    ├───────────────────────────┼───────────────────────────────┼────────────────────────────────┤")
    print("    │ Real-world Entity         │ Single point in latent space  │ Dynamic field with topology    │")
    print("    │ Interaction Mode          │ Inner product / Distance loss │ Boundary condition transfer    │")
    print("    │ Contradiction Handling    │ Irreversible weight overwrite │ Reaction tension & isolation   │")
    print("    │ State Recovery            │ Irreversible information loss │ Lossless SealedAttractor vault │")
    print("    └───────────────────────────┴───────────────────────────────┴────────────────────────────────┘")

    # 4. Causal Game Mechanics Engine Demonstration
    print("\n[4/4] Running Causal Game Mechanics Engine (Narrative Rupture & Recovery)...")
    game_engine = CausalGameMechanicsEngine()

    print("    - Action: Player assassinates King Arthur (Key Narrative NPC)...")
    game_res = game_engine.execute_player_action(action_type="ASSASSINATE", target_id="NPC_KING_ARTHUR")

    print(f"    - Status: {game_res['status']}")
    print(f"    - Quarantined Anomaly Zone: {game_res['anomaly_id']}")
    print(f"    - Emergent Mediating Node: {game_res['new_node_id']}")
    print(f"    - Updated Quest Rule: {game_res['new_rule']}")
    print(f"    - Restored Tension V_t: {game_res['final_tension']:.4f} (Under threshold 0.05)")

    print("\n==========================================================================")
    print("   Demonstration Completed Successfully!                                 ")
    print("==========================================================================")


if __name__ == "__main__":
    run_demo()
