#!/usr/bin/env python3
"""
demo_markov_blanket_interface.py

Demonstration of the Light Schnitt and Markov Blanket Interface.
Simulates cross-dimensional interaction between:
 1) SPATIAL_2D Grid (e.g., Image Canvas / Pixel Mesh)
 2) HIERARCHICAL_DAG Tree (e.g., JSON / AST Branching)

Visualizes phase alignment and state transition:
  GAS (floating/unbound) -> LIQUID (active boundary friction & alignment) -> ICE (Phase-Lock, FLOPs = 0)
"""

import math
import time
from core.sensory import (
    PhaseState,
    DimensionType,
    TopologyPointer,
    NativeTopology,
    MarkovBlanketInterface,
)


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def build_spatial_2d_topology() -> NativeTopology:
    """Builds a 2D Spatial Grid Native Topology (e.g., Image Grid Nodes)"""
    pointers = {
        "px_0_0": TopologyPointer(
            node_id="px_0_0",
            dimension=DimensionType.SPATIAL_2D,
            adjacent_ids={"px_0_1", "px_1_0"},
            phase_offset=0.0,
            frequency=1.0,
            flux_normal=(1.0, 0.0), # Outward right normal
        ),
        "px_0_1": TopologyPointer(
            node_id="px_0_1",
            dimension=DimensionType.SPATIAL_2D,
            adjacent_ids={"px_0_0", "px_1_1"},
            phase_offset=0.0,
            frequency=1.0,
            flux_normal=(1.0, 0.0),
        ),
    }
    boundary_nodes = {"px_0_0", "px_0_1"}
    return NativeTopology(
        topology_id="spatial_grid_2d",
        dimension=DimensionType.SPATIAL_2D,
        pointers=pointers,
        boundary_nodes=boundary_nodes,
    )


def build_hierarchical_dag_topology(
    phase_offset: float = math.pi,
    flux_normal: tuple = (1.0, 0.0),
    frequency: float = 3.0
) -> NativeTopology:
    """Builds a Hierarchical DAG Tree Native Topology (e.g., JSON Tree Nodes)"""
    pointers = {
        "json_root": TopologyPointer(
            node_id="json_root",
            dimension=DimensionType.HIERARCHICAL_DAG,
            adjacent_ids={"json_child1", "json_child2"},
            phase_offset=phase_offset,
            frequency=frequency,
            flux_normal=flux_normal,
        ),
        "json_child1": TopologyPointer(
            node_id="json_child1",
            dimension=DimensionType.HIERARCHICAL_DAG,
            adjacent_ids=set(),
            phase_offset=phase_offset,
            frequency=frequency,
            flux_normal=flux_normal,
        ),
    }
    boundary_nodes = {"json_root"}
    return NativeTopology(
        topology_id="hierarchical_tree_dag",
        dimension=DimensionType.HIERARCHICAL_DAG,
        pointers=pointers,
        boundary_nodes=boundary_nodes,
    )


def run_demo():
    print_banner("ELYASIA: MARKOV BLANKET INTERFACE & CROSS-DIMENSIONAL LIGHT SCHNITT DEMO")

    interface = MarkovBlanketInterface(friction_threshold=0.10)

    # ----------------------------------------------------
    # Stage 1: Initial Contact - Unaligned Phase (GAS)
    # ----------------------------------------------------
    print("\n[STAGE 1: Initial Boundary Contact - Floating Unbound State (GAS)]")
    spatial_topo = build_spatial_2d_topology()
    # Tree node pointing same direction (1.0, 0.0), mismatched frequency and phase
    dag_topo_unaligned = build_hierarchical_dag_topology(
        phase_offset=math.pi, flux_normal=(1.0, 0.0), frequency=5.0
    )

    state_gas, gas_data = interface.phase_transition_step(spatial_topo, dag_topo_unaligned)
    print(f" -> Current State: {state_gas.value}")
    print(f" -> Friction Output: {gas_data}")

    # ----------------------------------------------------
    # Stage 2: Boundary Friction & Phase Search (LIQUID)
    # ----------------------------------------------------
    print("\n[STAGE 2: Phase Alignment Search - Active Thermal Friction (LIQUID)]")
    # Tree node adjusting flux normal towards opposite direction (-0.9, 0.0) & phase aligning
    dag_topo_aligning = build_hierarchical_dag_topology(
        phase_offset=0.2, flux_normal=(-0.8, 0.0), frequency=1.5
    )

    schnitt_liquid = interface.execute_light_schnitt(spatial_topo, dag_topo_aligning)
    print(f" -> Light Schnitt Thermal Friction (Heat): {schnitt_liquid.thermal_friction:.4f}")
    print(f"    - Shear Friction (Dimensional Strain): {schnitt_liquid.shear_friction:.4f}")
    print(f"    - Flux Friction (Directional Distortion): {schnitt_liquid.flux_friction:.4f}")
    print(f"    - Temporal Friction (Phase/Freq Shift): {schnitt_liquid.temporal_friction:.4f}")
    print(f" -> Invariance Bridge (Sameness): {schnitt_liquid.invariance_bridge}")
    print(f" -> Phase Locked: {schnitt_liquid.phase_locked}")

    # ----------------------------------------------------
    # Stage 3: Phase-Lock & Direct Pointer Traversal (ICE)
    # ----------------------------------------------------
    print("\n[STAGE 3: Complete Phase Alignment & Phase-Lock Transition (ICE)]")
    # Perfectly facing flux normal (-1.0, 0.0), aligned phase and frequency
    dag_topo_aligned = build_hierarchical_dag_topology(
        phase_offset=0.01, flux_normal=(-1.0, 0.0), frequency=1.0
    )

    state_ice, ice_data = interface.phase_transition_step(spatial_topo, dag_topo_aligned)
    print(f" -> Current State: {state_ice.value}")
    print(f" -> Unified ICE Structure:")
    print(f"    - Source ID: {ice_data['source_id']}")
    print(f"    - Target ID: {ice_data['target_id']}")
    print(f"    - Invariance Bridge (Sameness Backbone): {ice_data['invariance_bridge']}")
    print(f"    - Divergence Axis (Difference Axes preserved): {ice_data['divergence_axis']}")
    print(f"    - Thermal Friction (Heat Dissipated): {ice_data['thermal_friction']:.6f}")
    print(f"    - Traversal Cost: {ice_data['traversal_cost']}")

    print_banner("DEMO COMPLETED SUCCESSFULLY - CROSS-DIMENSIONAL PHASE-LOCK ACHIEVED")


if __name__ == "__main__":
    run_demo()
