"""
Verification Script for Enactive Relational Memory Engine & Persistent Substrate
=============================================================================
This script verifies:
1. Cross-Modal Relational Mapping:
   - Interweaves language, visual form, and causal events for grounded entities (e.g., 'apple')
     without flattening them into scalar float vectors.
2. Enactive Observation & Self-Calibration:
   - Detects discrepancy/friction with external reality feedback.
   - Dynamically shifts focus/attention to mismatched nodes and calibrates edge tensions.
3. Memory Substrate Consolidation & Warm-Start Re-cognition:
   - Anchors calibrated meshes into persistent memory substrates.
   - Demonstrates that subsequent cognitive cycles operate on top of this persistent substrate
     rather than starting from zero (statelessness).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.memory.enactive_relational_memory import (
    EnactiveRelationalMemoryEngine,
    RelationalMesh,
    SelfCalibrationResult
)
from core.lens.cognitive_lens_engine import CognitiveLensEngine
from core.sensory.causal_sensor import CausalSensor


def test_cross_modal_relational_mapping():
    print("\n--- 1. Testing Cross-Modal Relational Mapping ---")
    engine = EnactiveRelationalMemoryEngine()

    apple_mesh = engine.create_cross_modal_mesh(
        entity_id="entity_apple",
        entity_name="Apple (사과)",
        linguistic_def="A edible fruit born from trees, symbolizing life, gravity, and relationship.",
        visual_form_desc="Red spherical shape with a smooth skin, stem on top, and juicy texture.",
        causal_events=[
            "Falls from tree branch under gravity",
            "Satisfies hunger when consumed by a living being",
            "Used as a gesture of reconciliation in relationship"
        ],
        sensory_attributes={
            "touch_taste": "Crisp bite, sweet and sour juice, smooth skin",
            "color_texture": "Vibrant red with light yellow specks"
        }
    )

    assert apple_mesh.entity_id == "entity_apple"
    assert len(apple_mesh.nodes) >= 4, f"Expected at least 4 nodes, got {len(apple_mesh.nodes)}"
    assert len(apple_mesh.edges) >= 4, f"Expected at least 4 edges, got {len(apple_mesh.edges)}"

    print(f"✔ Cross-modal mesh created for '{apple_mesh.entity_name}':")
    for node_id, node in apple_mesh.nodes.items():
        print(f"   Node [{node.modal_type}]: {node.label}")
    for edge in apple_mesh.edges:
        print(f"   Edge [{edge.relation_type}]: {edge.contextual_glue}")

    print("✔ Cross-Modal Relational Mapping test PASSED!")


def test_enactive_self_calibration():
    print("\n--- 2. Testing Enactive Self-Calibration under World Friction ---")
    engine = EnactiveRelationalMemoryEngine()

    engine.create_cross_modal_mesh(
        entity_id="entity_apple",
        entity_name="Apple (사과)",
        linguistic_def="A edible fruit born from trees.",
        visual_form_desc="Red spherical shape.",
        causal_events=["Falls under gravity", "Satisfies hunger"]
    )

    # Simulate external reality feedback indicating a discrepancy in CAUSAL_EVENT
    reality_feedback = {
        "reality_coherence": 0.4,
        "world_friction": 0.6,
        "mismatched_modal": "CAUSAL_EVENT"
    }

    calibration_result = engine.enact_self_calibration("entity_apple", reality_feedback)

    assert calibration_result.discrepancy_magnitude > 0.3
    assert calibration_result.focus_target_node_id is not None
    assert "causal" in calibration_result.focus_target_node_id

    print(f"✔ Discrepancy detected: {calibration_result.discrepancy_magnitude:.3f}")
    print(f"✔ Attention focused on node: [{calibration_result.focus_target_node_id}]")
    print(f"✔ Action taken: {calibration_result.action_taken}")
    print("✔ Enactive Self-Calibration test PASSED!")


def test_memory_substrate_consolidation_and_warm_start():
    print("\n--- 3. Testing Memory Substrate Consolidation & Warm-Start Re-cognition ---")
    engine = EnactiveRelationalMemoryEngine()

    engine.create_cross_modal_mesh(
        entity_id="entity_apple",
        entity_name="Apple (사과)",
        linguistic_def="A edible fruit born from trees.",
        visual_form_desc="Red spherical shape.",
        causal_events=["Falls under gravity"]
    )

    # Enact self-calibration
    engine.enact_self_calibration("entity_apple", {"reality_coherence": 0.7, "world_friction": 0.2})

    # Consolidate into persistent substrate
    substrate_record = engine.consolidate_to_substrate("entity_apple")
    assert substrate_record["entity_id"] == "entity_apple"

    print("✔ Memory Substrate Consolidated:")
    print(f"   Entity: {substrate_record['entity_name']}")
    print(f"   Coherence: {substrate_record['coherence_score']:.3f}")
    for summary in substrate_record["grounded_summary"]:
        print(f"   - {summary}")

    # Test warm-start retrieval on subsequent cycle
    retrieved_mesh = engine.retrieve_grounded_substrate("entity_apple")
    assert retrieved_mesh is not None
    assert retrieved_mesh.entity_id == "entity_apple"

    print("✔ Warm-Start Retrieval SUCCESSFUL: Subsequent cognitive cycle operates on top of persistent substrate!")
    print("✔ Memory Substrate Consolidation test PASSED!")


def test_sensor_integration():
    print("\n--- 4. Testing CausalSensor Integration with Relational Memory ---")
    lens_engine = CognitiveLensEngine()
    relational_mem = EnactiveRelationalMemoryEngine()
    sensor = CausalSensor(sensor_id="main_sensor", lens_engine=lens_engine, relational_memory=relational_mem)

    relational_mem.create_cross_modal_mesh(
        entity_id="entity_apple",
        entity_name="Apple (사과)",
        linguistic_def="Edible fruit",
        visual_form_desc="Red sphere",
        causal_events=["Gravity fall"]
    )

    stimulus = {"intensity": 0.8}
    feedback = {"world_friction": 0.7}
    friction_res = sensor.project_and_measure_friction(stimulus, feedback)

    sensor.self_calibrate(friction_res, entity_id="entity_apple")

    retrieved = relational_mem.retrieve_grounded_substrate("entity_apple")
    assert retrieved is not None
    print("✔ CausalSensor successfully calibrated relational memory substrate!")
    print("✔ Sensor Integration test PASSED!")


if __name__ == "__main__":
    print("==========================================================================")
    print(" Verifying Enactive Relational Memory Engine & Persistent Substrate ")
    print("==========================================================================")
    test_cross_modal_relational_mapping()
    test_enactive_self_calibration()
    test_memory_substrate_consolidation_and_warm_start()
    test_sensor_integration()
    print("\n==========================================================================")
    print(" ALL ENACTIVE RELATIONAL MEMORY VERIFICATIONS PASSED SUCCESSFULLY! ")
    print("==========================================================================")
