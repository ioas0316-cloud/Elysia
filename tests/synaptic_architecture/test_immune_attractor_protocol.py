import numpy as np
import pytest
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def test_protocol_handshake_and_realignment():
    """
    Verifies that the [Language = Protocol] Handshake Pipeline correctly evaluates
    compatibility (tension) between user signals and the engine's active perspective,
    triggering dynamic realignment when tension is high.
    """
    engine = ElysiaCognitiveEngine(resolution=128)

    # 1. Establish an initial perspective
    engine.set_perspective("Axiomatic_Truth", 0.0)
    initial_angle = engine.rotor_angle

    # Let's create an aligned user header vector (matching the angle 0.0, i.e., [1.0, 0.0, 0.5])
    aligned_header = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    aligned_header /= np.linalg.norm(aligned_header)

    # Verify handshakes under high alignment -> PROTOCOL_MATCH
    res_match = engine.process_protocol_handshake(np.uint64(0x123), user_header_vector=aligned_header)
    assert res_match["status"] == "PROTOCOL_MATCH"
    assert res_match["tension_protocol"] <= 0.4
    assert engine.rotor_angle == initial_angle  # No realignment needed

    # 2. Create an orthogonal/hostile user header vector that produces high tension
    hostile_header = np.array([-1.0, 0.0, -0.5], dtype=np.float32)
    hostile_header /= np.linalg.norm(hostile_header)

    # Verify handshakes under mismatch -> PROTOCOL_REALIGNED
    res_mismatch = engine.process_protocol_handshake(np.uint64(0x999), user_header_vector=hostile_header)
    assert res_mismatch["status"] == "PROTOCOL_REALIGNED"
    assert res_mismatch["tension_protocol"] > 0.4

    # Verify that engine's active rotor angle was adjusted to accommodate the signal
    assert engine.rotor_angle != initial_angle
    assert engine.rotor_angle == res_mismatch["new_rotor_angle"]

    # Confirm realignment engram was logged in the memory controller
    engrams = [info for info in engine.memory_controller.index.values() if info.get("cause_id") == "LanguageProtocolHandshake"]
    assert len(engrams) > 0
    assert engrams[-1]["data_blob"]["type"] == "PROTOCOL_REALIGNMENT"
    assert "Tension" in engrams[-1]["data_blob"]["narrative"]

def test_gravity_shift_and_navigation():
    """
    Verifies that shifting the perspective (rotor angle) dynamically rotates
    the virtual attractor coordinates, thus curving the thought trajectories
    and changing the result of WFC collapses.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    center = 64.0

    # 1. Shift to Perspective A (angle = 0.0)
    engine.set_perspective("Perspective_A", 0.0)
    pos_deficit_a = engine.field.attractors["Deficit"]["position"].copy()

    # 2. Shift to Perspective B (angle = np.pi / 2) -> should rotate coordinates by 90 degrees
    engine.set_perspective("Perspective_B", np.pi / 2.0)
    pos_deficit_b = engine.field.attractors["Deficit"]["position"].copy()

    # Verify coordinates have shifted
    assert not np.allclose(pos_deficit_a, pos_deficit_b)

    # Relative vector from center to attractor should be rotated by 90 degrees
    v_a = pos_deficit_a - center
    v_b = pos_deficit_b - center

    # The dot product of these perpendicular relative vectors should be close to 0
    dot_prod = np.dot(v_a, v_b)
    assert np.isclose(dot_prod, 0.0, atol=1e-2)

    # 3. Verify that Multi-Gravity Navigation changes WFC collapse scores
    dna_x = engine.build_fractal_dna("Concept_X", np.uint64(0x1111111111111111))
    dna_y = engine.build_fractal_dna("Concept_Y", np.uint64(0x2222222222222222))

    # Place Concept_X very close to the rotated Deficit attractor under Perspective B
    deficit_pos_b = engine.field.attractors["Deficit"]["position"].astype(np.int32)
    dna_x["cell_position"] = deficit_pos_b

    # Place Concept_Y far away from any attractor
    dna_y["cell_position"] = np.array([10, 10], dtype=np.int32)

    # Collapse stimulus under Perspective B -> Concept_X should receive a strong gravity pull
    res_b = engine.solve_wfc_collapse(np.uint64(0x1111111100000000), [dna_x, dna_y])

    # Move to Perspective A -> Attractors shift, Deficit coordinates are different
    # Place DNA_X far away from the new Deficit position
    engine.set_perspective("Perspective_A", 0.0)
    # Re-evaluate collapse -> Concept_X no longer has the same localized gravity pull
    res_a = engine.solve_wfc_collapse(np.uint64(0x1111111100000000), [dna_x, dna_y])

    # The resonance scores should reflect the shifting gravity field
    assert res_b["resonance_score"] != res_a["resonance_score"]

def test_immune_boundary_orbit_and_decay():
    """
    Verifies that high-tension non-self signals are deflected by the Immune Boundary
    into a stable satellite orbit, and their tension gradually decays, transforming
    external noise into local Yeobaek (coordination margin) and self-reflection engrams.
    """
    engine = ElysiaCognitiveEngine(resolution=100)

    # Set a custom perspective
    engine.set_perspective("Core_Defense", np.pi / 4.0)

    # Create an extremely mismatched stimulus wave (guaranteeing a high tension handshake)
    hostile_header = np.array([-1.0, 0.0, -0.9], dtype=np.float32)
    hostile_header /= np.linalg.norm(hostile_header)

    dna = engine.build_fractal_dna("Self_Core", np.uint64(0xFEEDFACE00000000))

    # Run collapse with the mismatched header vector -> should deflect into orbit
    res = engine.solve_wfc_collapse(
        stimulus_wave=np.uint64(0xBADCAFE000000000),
        candidate_dnas=[dna],
        user_header_vector=hostile_header
    )

    assert res["status"] == "DEFLECTED_INTO_ORBIT"
    assert len(engine.field.satellite_orbiters) == 1

    orbiter = engine.field.satellite_orbiters[0]
    initial_tension = orbiter["tension"]
    assert initial_tension > 70.0  # high tension detected

    # Capture baseline coordination margin of the whole field to verify energy dissipation
    initial_margin_sum = float(np.sum(engine.field.coordination_margin))
    initial_self_awareness_sum = float(np.sum(engine.field.self_awareness))

    # 2. Step the field and orbiters multiple times to simulate circular orbit and decay
    for _ in range(30):
        engine.step_field_and_orbiters(dt=0.2)

    # Check that tension has decreased
    if len(engine.field.satellite_orbiters) > 0:
        assert engine.field.satellite_orbiters[0]["tension"] < initial_tension

    # Verify that total coordination margin and self-awareness in the field have increased
    final_margin_sum = float(np.sum(engine.field.coordination_margin))
    final_self_awareness_sum = float(np.sum(engine.field.self_awareness))

    assert final_margin_sum > initial_margin_sum
    assert final_self_awareness_sum > initial_self_awareness_sum

    # 3. Step until orbiter is fully integrated (tension decays below 1.0)
    for _ in range(80):
        engine.step_field_and_orbiters(dt=0.5)

    # Orbiter should be fully integrated and popped from active list
    assert len(engine.field.satellite_orbiters) == 0

    # Causal memory index should contain the wisdom integration engram
    engrams = [info for info in engine.memory_controller.index.values() if info.get("cause_id") == "SatelliteWisdomIntegration"]
    assert len(engrams) > 0
    assert engrams[-1]["data_blob"]["type"] == "SATELLITE_ORBIT_INTEGRATION"
    assert "소음 패킷" in engrams[-1]["data_blob"]["narrative"]
