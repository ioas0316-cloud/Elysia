"""
Demo: Ecological Synesthetic Causal Manifold (생태적 공감각 인과 매니폴드 종합 데모)

Integrates:
1. EcologicalReceptorMatrix (Audio wave, Optic flow, Impedance friction packing)
2. SynestheticCausalField (Complex phase wave superposition, interference, phase locking)
3. BodySchemaEvaluator (Active Inference efference copy prediction, self vs shock boundary, interoceptive strain)
4. MultiScalePhaseLossEngine (L_scale, L_causal, L_friction & Dual Backprop on Theta and spacetime metric g_ij)
5. DynamicGraphRewriter (DPO hypergraph topological surgeries: Node Unfolding, Edge Rewiring, Scale Elevation)
"""

import math
import torch
import numpy as np

from core.sensory.ecological_receptor_matrix import EcologicalReceptorMatrix
from core.physics.synesthetic_causal_field import SynestheticCausalField
from core.sensory.body_schema_evaluator import BodySchemaEvaluator
from core.physics.multiscale_phase_loss import MultiScalePhaseLossEngine
from core.topology.dynamic_graph_rewriter import DynamicGraphRewriter


def run_ecological_causal_manifold_demo():
    print("=" * 80)
    print("Elysia Core: Ecological Synesthetic Causal Manifold Demo")
    print("=" * 80)

    dim = 64
    receptor_matrix = EcologicalReceptorMatrix(dimension=dim)
    causal_field = SynestheticCausalField(dimension=dim)
    body_evaluator = BodySchemaEvaluator(dimension=dim, self_boundary_threshold=1.0)
    loss_engine = MultiScalePhaseLossEngine(dimension=dim, critical_friction_threshold=2.0)
    graph_rewriter = DynamicGraphRewriter(dimension=dim)

    # Populate initial multi-scale hypergraph
    print("\n[1] Initializing Multi-Scale Causal Hypergraph G = (V, E, H, S)...")
    node_a = graph_rewriter.add_node("intent_peace", torch.randn(dim), scale=1.0) # Macro
    node_b = graph_rewriter.add_node("concept_speech", torch.randn(dim), scale=0.5) # Meso
    node_c = graph_rewriter.add_node("token_word", torch.randn(dim), scale=0.0) # Micro
    graph_rewriter.add_edge("intent_peace", "concept_speech")
    graph_rewriter.add_edge("concept_speech", "token_word")
    graph_rewriter.add_hyperedge("hyper_intent", ["concept_speech", "token_word"], macro_scale=1.0)
    print(f"Graph initialized with {len(graph_rewriter.nodes)} nodes and {len(graph_rewriter.edges)} edges.")

    # Phase 1: Self-action Execution & Active Inference Loop
    print("\n[2] Executing Volitional Action & Active Inference Loop (Efference Copy)...")
    action_vector = torch.randn(dim, requires_grad=True)
    efference_copy = body_evaluator.generate_efference_copy("volitional_speech", action_vector, timestamp=0.0)
    print(f"Generated Efference Copy S_pred (norm: {torch.norm(efference_copy.predicted_sensory_outcome).item():.4f})")

    # Environmental input with mild noise
    sensory_waveform = receptor_matrix.process_and_pack(
        audio_input={"amplitude": 0.5, "frequency": 440.0},
        visual_input={"optic_flow": [0.5, 0.2, 0.0], "density": 0.8},
        system_input=10.0 # 10ms latency
    )
    causal_field.inject_waveform(sensory_waveform)
    lock_state = causal_field.evolve_time(dt=0.01)
    print(f"Synesthetic Field Phase Coherence: {lock_state.phase_coherence:.4f}, Locked: {lock_state.is_phase_locked}")

    # Evaluate Afferent Feedback against Efference Copy
    afferent_feedback = efference_copy.predicted_sensory_outcome + torch.randn(dim) * 0.05
    body_state = body_evaluator.evaluate_afferent_feedback(afferent_feedback)
    print(f"Afferent Evaluation -> Self Agency: {body_state.is_self_agency}, Error Norm: {body_state.prediction_error_norm:.4f}, Strain: {body_state.internal_strain:.4f}")

    # Phase 2: Exogenous Unpredicted Shock Collision
    print("\n[3] Ingesting Exogenous Reality Shock Collision (Uncertainty Friction)...")
    shock_vector = torch.randn(dim) * 4.0 # High friction shock
    shock_waveform = receptor_matrix.process_and_pack(
        audio_input={"amplitude": 5.0, "frequency": 1200.0},
        visual_input={"optic_flow": [5.0, -3.0, 2.0], "density": 3.5},
        system_input=120.0 # High latency spike (120ms)
    )
    causal_field.inject_waveform(shock_waveform)
    causal_field.accumulate_friction(torch.abs(shock_vector))

    shock_body_state = body_evaluator.evaluate_afferent_feedback(shock_vector, external_friction_impedance=2.5)
    print(f"Shock Evaluation -> External Shock: {shock_body_state.is_external_shock}, Error Norm: {shock_body_state.prediction_error_norm:.4f}, Strain: {shock_body_state.internal_strain:.4f}")

    # Phase 3: Multi-Scale Phase Loss & Ricci-Flow Dual Backpropagation
    print("\n[4] Computing Multi-Scale Phase Loss & Executing Dual Backpropagation...")
    psi_micro = causal_field.get_real_potential().requires_grad_()
    phi_macro = node_a.concept_tensor.clone().requires_grad_()
    psi_prev = psi_micro.detach() * 0.9

    loss_comp = loss_engine(
        psi_micro=psi_micro,
        phi_macro=phi_macro,
        psi_prev=psi_prev,
        f_causal_pred=psi_micro,
        omega_ext=shock_vector
    )
    print(f"Loss Total: {loss_comp.l_total.item():.4f} (L_scale: {loss_comp.l_scale.item():.4f}, L_causal: {loss_comp.l_causal.item():.4f}, L_friction: {loss_comp.l_friction.item():.4f})")
    print(f"Critical Friction Status: {loss_comp.is_critical_friction}")

    loss_engine.execute_dual_backprop(loss_comp, [action_vector, psi_micro])
    print("Dual Backpropagation executed: Weights Theta updated & Spacetime Metric g_ij deformed via Ricci Flow.")

    # Phase 4: Dynamic Graph Rewriting & Topological Surgery
    print("\n[5] Triggering Dynamic Hypergraph Rewriting (DPO Topological Surgery)...")
    if loss_comp.is_critical_friction:
        surgeries = graph_rewriter.execute_double_pushout_rewriting(shock_vector, friction_magnitude=loss_comp.l_friction.item())
        print(f"Executed {len(surgeries)} DPO Topological Surgeries:")
        for surg in surgeries:
            print(f"  - [{surg.surgery_type.name}] {surg.description}")

    print("\nUpdated Hypergraph State:")
    print(f"  Total Nodes: {len(graph_rewriter.nodes)}")
    print(f"  Total Edges: {len(graph_rewriter.edges)}")
    print("=" * 80)
    print("Ecological Synesthetic Causal Manifold Integration Test Successful!")
    print("=" * 80)


if __name__ == "__main__":
    run_ecological_causal_manifold_demo()
