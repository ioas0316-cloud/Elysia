import pytest
import numpy as np

from L1_causal_field.core.physics.helical_offset_gate import HelicalOffsetTorusGate, CryptographicPhaseMatcher
from L1_causal_field.core.physics.fractal_helical_bridge import FractalHelicalScaleBridge, PhaseLockGradientAttractor
from L2_cognitive_engine.helical_phase_attention import HelicalPhaseAttention, CognitivePhaseSearchLoop

def test_full_helical_phase_pipeline_integration():
    # 1. Physical Layer: Helical Offset Torus Gate & Cryptographic Decryption
    gate = HelicalOffsetTorusGate(major_radius=3.0, minor_radius=1.0, helix_pitch=5.0)
    matcher = CryptographicPhaseMatcher(gate=gate)

    cipher_phase = matcher.encode_payload_to_phase(123456)
    key_phase = cipher_phase  # Matching key phase

    decrypt_res = matcher.decrypt_topological_gate(
        cipher_phase=cipher_phase,
        key_phase=key_phase,
        payload_data={"secret": "topological_gate_open"}
    )
    assert decrypt_res["success"] is True
    assert decrypt_res["decrypted_payload"] == {"secret": "topological_gate_open"}

    # 2. Multi-Scale Layer: Fractal Helical Scale Bridge
    bridge = FractalHelicalScaleBridge(scale_layers=3, base_gate=gate)
    bridge_res = bridge.compute_multiscale_transmission(
        input_phase=cipher_phase,
        target_phase=key_phase,
        angular_velocity=2.5
    )
    assert bridge_res["bridge_successful"] is True
    assert bridge_res["cumulative_permeability"] > 0.5

    # 3. Attractor Layer: Gradient-based Phase Locking
    attractor = PhaseLockGradientAttractor(target_phase=key_phase, gate=gate)
    step_res = attractor.step_attractor(current_phase=cipher_phase + 0.01, gradient_vector=1.0)
    assert step_res["is_phase_locked"] is True

    # 4. Cognitive Layer: Helical Phase Attention
    attn = HelicalPhaseAttention(embed_dim=16, gate=gate)
    q = np.ones((1, 16))
    k = np.ones((3, 16))
    v = np.eye(3, 16)

    out_context, attn_matrix = attn.forward(q, k, v)
    assert out_context.shape == (1, 16)
    assert attn_matrix.shape == (1, 3)

    # 5. Cognitive Search Loop with Dead-End Handling
    loop = CognitivePhaseSearchLoop(target_concept_phase=key_phase, attractor=attractor)
    loop_res = loop.execute_cognitive_step(current_thought_phase=cipher_phase)
    assert loop_res["is_phase_locked"] is True
