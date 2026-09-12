"""
Unit tests for SpatioTemporalSyncAdapter and ReactiveDAG cascade.
"""

import pytest
from core.topology.spatiotemporal_reactive_cascade import (
    AudioFrame,
    DAGNode,
    ReactiveDAG,
    SemanticState,
    SpatioTemporalSyncAdapter,
    VisualFrame,
    action_decision_operator,
    causal_integrity_operator,
    sync_input_operator,
)


def test_spatiotemporal_sync_adapter_normal():
    adapter = SpatioTemporalSyncAdapter()
    v_frame = VisualFrame(timestamp_ms=100.0, bbox=(0.4, 0.2, 0.6, 0.8), label="speaker")
    a_frame = AudioFrame(timestamp_ms=105.0, doa_angle_deg=180.0, decibel=75.0, sound_type="speech")

    bindings = adapter.process_sync(v_frame, a_frame)
    assert bindings["TEMPORAL_SYNC"] == "PASS"
    assert bindings["SPATIAL_SYNC"] == "PASS"
    assert bindings["CAUSAL_INTEGRITY"] == "COHERENT_SINGLE_SOURCE"
    assert bindings["SYNC_STATUS"] == "VERIFIED"


def test_spatiotemporal_sync_adapter_spatial_anomaly():
    adapter = SpatioTemporalSyncAdapter()
    v_frame = VisualFrame(timestamp_ms=100.0, bbox=(0.1, 0.2, 0.2, 0.8), label="speaker")  # ~153 deg
    a_frame = AudioFrame(timestamp_ms=105.0, doa_angle_deg=220.0, decibel=80.0, sound_type="speech")

    bindings = adapter.process_sync(v_frame, a_frame)
    assert bindings["TEMPORAL_SYNC"] == "PASS"
    assert bindings["SPATIAL_SYNC"] == "FAIL"
    assert bindings["CAUSAL_INTEGRITY"] == "OFF_SCREEN_OR_DUBBED"
    assert bindings["SYNC_STATUS"] == "SPATIAL_ANOMALY"


def test_reactive_dag_cascade_scenario1():
    dag = ReactiveDAG()
    adapter = SpatioTemporalSyncAdapter()

    sync_node = DAGNode("Sync_Input_Node", lambda inputs: None)
    causal_node = DAGNode("Causal_Integrity_Node", causal_integrity_operator)
    action_node = DAGNode("Action_Decision_Node", action_decision_operator)

    dag.add_node(sync_node)
    dag.add_node(causal_node)
    dag.add_node(action_node)

    dag.add_edge("Sync_Input_Node", "Causal_Integrity_Node")
    dag.add_edge("Causal_Integrity_Node", "Action_Decision_Node")

    v_frame = VisualFrame(timestamp_ms=100.0, bbox=(0.4, 0.2, 0.6, 0.8), label="speaker")
    a_frame = AudioFrame(timestamp_ms=105.0, doa_angle_deg=180.0, decibel=75.0, sound_type="speech")

    initial_state = sync_input_operator(adapter, v_frame, a_frame)
    dag.propagate("Sync_Input_Node", initial_state)

    assert "HIGH_CAUSAL_CONFIDENCE" in dag.nodes["Causal_Integrity_Node"].state.qualities
    assert dag.nodes["Causal_Integrity_Node"].state.relational_bindings["AGENT_ATTENTION"] == "FOCUS_TARGET"
    assert "ACTION_TRACKING" in dag.nodes["Action_Decision_Node"].state.qualities
    assert dag.nodes["Action_Decision_Node"].state.relational_bindings["EXECUTE_COMMAND"] == "LOCK_CAMERA_AND_LISTEN"


def test_reactive_dag_cascade_scenario2():
    dag = ReactiveDAG()
    adapter = SpatioTemporalSyncAdapter()

    sync_node = DAGNode("Sync_Input_Node", lambda inputs: None)
    causal_node = DAGNode("Causal_Integrity_Node", causal_integrity_operator)
    action_node = DAGNode("Action_Decision_Node", action_decision_operator)

    dag.add_node(sync_node)
    dag.add_node(causal_node)
    dag.add_node(action_node)

    dag.add_edge("Sync_Input_Node", "Causal_Integrity_Node")
    dag.add_edge("Causal_Integrity_Node", "Action_Decision_Node")

    v_frame = VisualFrame(timestamp_ms=100.0, bbox=(0.1, 0.2, 0.2, 0.8), label="speaker")
    a_frame = AudioFrame(timestamp_ms=105.0, doa_angle_deg=220.0, decibel=80.0, sound_type="speech")

    anomalous_state = sync_input_operator(adapter, v_frame, a_frame)
    dag.propagate("Sync_Input_Node", anomalous_state)

    assert "OUT_OF_FRAME_ATTENTION" in dag.nodes["Causal_Integrity_Node"].state.qualities
    assert dag.nodes["Causal_Integrity_Node"].state.relational_bindings["AGENT_ATTENTION"] == "SCAN_SURROUNDINGS"
    assert "ACTION_SEARCH" in dag.nodes["Action_Decision_Node"].state.qualities
    assert dag.nodes["Action_Decision_Node"].state.relational_bindings["EXECUTE_COMMAND"] == "PAN_CAMERA_TO_DOA"
