import pytest
import math
from core.sensory.markov_blanket_interface import (
    MarkovBlanketInterface,
    NativeTopology,
    TopologyPointer,
    DimensionType,
    PhaseState,
)


def test_aligned_pointers_phase_lock_ice():
    interface = MarkovBlanketInterface(friction_threshold=0.15)

    # Source: 2D Spatial Grid Boundary Node
    s_ptr = TopologyPointer(
        node_id="grid_0_0",
        dimension=DimensionType.SPATIAL_2D,
        adjacent_ids={"grid_0_1", "grid_1_0"},
        phase_offset=0.1,
        frequency=1.0,
        flux_normal=(1.0, 0.0)
    )
    source = NativeTopology(
        topology_id="spatial_grid_src",
        dimension=DimensionType.SPATIAL_2D,
        pointers={"grid_0_0": s_ptr},
        boundary_nodes={"grid_0_0"}
    )

    # Target: DAG Tree Boundary Node with aligned phase & opposing flux normal
    t_ptr = TopologyPointer(
        node_id="tree_root",
        dimension=DimensionType.HIERARCHICAL_DAG,
        adjacent_ids={"tree_child_1", "tree_child_2"},
        phase_offset=0.12,
        frequency=1.0,
        flux_normal=(-1.0, 0.0)  # Facing opposite -> low flux friction
    )
    target = NativeTopology(
        topology_id="hierarchical_dag_tgt",
        dimension=DimensionType.HIERARCHICAL_DAG,
        pointers={"tree_root": t_ptr},
        boundary_nodes={"tree_root"}
    )

    state, result = interface.phase_transition_step(source, target)

    assert state == PhaseState.ICE
    assert result["phase_state"] == PhaseState.ICE
    assert result["invariance_bridge"] == {"grid_0_0": "tree_root"}
    assert result["divergence_axis"] == (DimensionType.SPATIAL_2D, DimensionType.HIERARCHICAL_DAG)
    assert result["traversal_cost"] == "FLOPs = 0 (Direct Graph Pointer Traversal)"


def test_misaligned_pointers_liquid_state():
    interface = MarkovBlanketInterface(friction_threshold=0.05)  # Strict threshold

    s_ptr = TopologyPointer(
        node_id="grid_0_0",
        dimension=DimensionType.SPATIAL_2D,
        adjacent_ids={"grid_0_1", "grid_1_0", "grid_1_1", "grid_2_0"},  # High degree
        phase_offset=0.0,
        frequency=1.0,
        flux_normal=(1.0, 0.0)
    )
    source = NativeTopology(
        topology_id="src",
        dimension=DimensionType.SPATIAL_2D,
        pointers={"grid_0_0": s_ptr},
        boundary_nodes={"grid_0_0"}
    )

    t_ptr = TopologyPointer(
        node_id="tree_root",
        dimension=DimensionType.HIERARCHICAL_DAG,
        adjacent_ids=set(),  # Degree 0 -> High shear friction
        phase_offset=math.pi, # 180 deg out of phase -> High temporal friction
        frequency=2.5,
        flux_normal=(1.0, 0.0) # Same flux direction -> High flux friction
    )
    target = NativeTopology(
        topology_id="tgt",
        dimension=DimensionType.HIERARCHICAL_DAG,
        pointers={"tree_root": t_ptr},
        boundary_nodes={"tree_root"}
    )

    state, result = interface.phase_transition_step(source, target)

    assert state in (PhaseState.LIQUID, PhaseState.GAS)
    assert result["friction"] > 0.05


def test_unbound_pointers_gas_state():
    interface = MarkovBlanketInterface(friction_threshold=0.01)

    source = NativeTopology(
        topology_id="src_empty",
        dimension=DimensionType.SPATIAL_2D,
        pointers={},
        boundary_nodes=set()
    )
    target = NativeTopology(
        topology_id="tgt_empty",
        dimension=DimensionType.HIERARCHICAL_DAG,
        pointers={},
        boundary_nodes=set()
    )

    state, result = interface.phase_transition_step(source, target)
    assert state == PhaseState.GAS
