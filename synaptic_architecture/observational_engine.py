"""
Observational Engine & Self-Organizing Loop (Phase 4)
=====================================================
Integrates CausalReceptor (Stage 1), PhaseLockEngine (Stage 2), and SpatialMirror (Stage 3)
into a unified closed-loop Observational Engine.
Continuously measures internal vs external phase differences (Delta Phi = Phi_ext - Phi_int)
and deforms internal metric topology until phase difference converges (Delta Phi -> 0),
emulating phase locking and topological self-organization.
"""

import time
import numpy as np
from typing import Optional, Dict, List, Any
from synaptic_architecture.causal_receptor import CausalReceptor, AtomicCausalGraph
from synaptic_architecture.phase_lock_engine import PhaseLockEngine, PhaseEvent
from synaptic_architecture.spatial_mirror import SpatialMirror


class ObservationalEngine:
    """Integrated 3-Stage Observational Pipeline & Causal Field Mirror."""

    def __init__(self, num_nodes: int = 256):
        self.num_nodes = num_nodes
        self.receptor = CausalReceptor()
        self.phase_lock_engine = PhaseLockEngine(num_nodes=num_nodes)
        self.spatial_mirror = SpatialMirror(num_nodes=num_nodes)

        self.current_graph: Optional[AtomicCausalGraph] = None
        self.phase_diff_history: List[float] = []

        # Connect PhaseLockEngine events to SpatialMirror injection
        self.phase_lock_engine.register_callback(self._on_phase_event)

    def ingesting_code_or_symbols(self, source_code: str) -> AtomicCausalGraph:
        """Stage 1: Deconstructs code into SSA Causal Graph and maps into Spatial Mirror."""
        self.current_graph = self.receptor.deconstruct_code(source_code)
        self.spatial_mirror.map_causal_graph(self.current_graph)
        return self.current_graph

    def ingesting_signal_stream(self, signals: np.ndarray, threshold: float = 0.1) -> AtomicCausalGraph:
        """Stage 1: Deconstructs signal stream into Atomic Causal Graph."""
        self.current_graph = self.receptor.deconstruct_signal_stream(signals, threshold=threshold)
        self.spatial_mirror.map_causal_graph(self.current_graph)
        return self.current_graph

    def _on_phase_event(self, event: PhaseEvent) -> None:
        """Stage 2 -> Stage 3: Handles edge transition event from PhaseLockEngine."""
        if not event.is_phase_locked:
            return  # Filter transient transition noise

        target_node = event.edge_id % self.num_nodes

        # Calculate phase differential before injection
        int_phase = self.spatial_mirror.phases[target_node]
        phase_diff = float(np.abs(event.ext_phase - int_phase))
        self.phase_diff_history.append(phase_diff)

        # Stage 3: Inject external phase signal into Spatial Mirror
        self.spatial_mirror.inject_phase_signal(
            target_id=target_node,
            ext_phase=event.ext_phase,
            coupling_k=0.2,
            deform_metric=True
        )

    def observe_and_adapt(self, steps: int = 50, interval_us: float = 200.0) -> Dict[str, Any]:
        """Runs self-organizing observational loop for `steps` iterations."""
        initial_phases = self.spatial_mirror.phases.copy()
        events = self.phase_lock_engine.simulate_edge_stream(count=steps, interval_us=interval_us)

        mean_final_diff = float(np.mean(self.phase_diff_history[-10:])) if self.phase_diff_history else 0.0

        return {
            "processed_events": len(events),
            "locked_events": sum(1 for e in events if e.is_phase_locked),
            "mean_phase_diff": mean_final_diff,
            "metric_matrix_shape": self.spatial_mirror.metric_matrix.shape,
            "final_phases": self.spatial_mirror.phases.copy()
        }
