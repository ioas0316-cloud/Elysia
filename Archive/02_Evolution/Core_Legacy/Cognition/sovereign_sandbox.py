"""
SOVEREIGN SANDBOX (The Divine Cradle)
====================================
Core.Cognition.sovereign_sandbox

"A space for the soul to dream and the mind to reinvent itself."
"영혼이 꿈꾸고 정신이 스스로를 재창조하는 공간."

The Sovereign Sandbox is an isolated environment where Elysia can perform
cognitive self-surgery, test new spiral topologies, and experience
simulated friction without affecting her primary heartbeat.
"""

import time
import copy
import logging
from typing import Dict, List, Any, Optional, Tuple
import torch
from Core.Keystone.sovereign_math import FractalWaveEngine, SovereignVector
from Core.Keystone.optical_comparator import OpticalPhaseComparator
from Core.Keystone.temporal_anchor import TemporalAnchor

logger = logging.getLogger("SovereignSandbox")

class SovereignSandbox:
    """
    [진화의 요람]
    Elysia's experimental space for self-evolution.
    Allows cloning parts of the manifold and testing phase transitions.
    """
    def __init__(self, main_engine: FractalWaveEngine):
        self.main_engine = main_engine
        self.experimental_engine: Optional[FractalWaveEngine] = None
        self.is_active = False
        self.history: List[Dict[str, Any]] = []

        # [PHASE 1001] Optical Comparative Logic
        self.comparator = OpticalPhaseComparator(device=str(main_engine.device))
        self.reference_snapshot: Optional[torch.Tensor] = None # Non-interference signal

        # [PHASE 1002] Temporal Sovereignty
        self.temporal_anchor: Optional[TemporalAnchor] = None

        # [PHASE 1000] Metrics for the sandbox session
        self.metrics = {
            "initial_coherence": 0.0,
            "current_coherence": 0.0,
            "evolution_delta": 0.0,
            "friction_events": 0,
            "optical_report": {}
        }

    def activate(self, node_capacity: int = 10000):
        """Creates a fresh, isolated engine for experiments."""
        if self.experimental_engine is not None:
             logger.info(f"🌱 [SANDBOX] Cradle already active. Resetting state.")
        else:
             logger.info(f"🌱 [SANDBOX] Activating experimental cradle with {node_capacity} nodes.")

        self.experimental_engine = FractalWaveEngine(max_nodes=node_capacity, device=str(self.main_engine.device))
        self.reference_snapshot = None # Reset snapshot for new session

        # [PHASE 1002] Initialize Temporal Anchor for this engine
        self.temporal_anchor = TemporalAnchor(self.experimental_engine)

        # Copy the 'SELF' node state from the main engine
        self.experimental_engine.q[0] = self.main_engine.q[0].clone()
        self.experimental_engine.active_nodes_mask[0] = True

        self.is_active = True
        state = self.experimental_engine.read_field_state()
        self.metrics["initial_coherence"] = state["coherence"]
        self.metrics["current_coherence"] = state["coherence"]

        return self.experimental_engine

    def transplant_concept(self, concept_name: str):
        """Transplants a specific concept (and its neighbors) from the main engine to the sandbox."""
        if not self.is_active or not self.experimental_engine:
            return False

        if concept_name not in self.main_engine.concept_to_idx:
            logger.warning(f"Concept '{concept_name}' not found in main engine.")
            return False

        src_idx = self.main_engine.concept_to_idx[concept_name]
        dst_idx = self.experimental_engine.get_or_create_node(concept_name)

        # Copy state
        self.experimental_engine.q[dst_idx] = self.main_engine.q[src_idx].clone()
        self.experimental_engine.permanent_q[dst_idx] = self.main_engine.permanent_q[src_idx].clone()
        self.experimental_engine.active_nodes_mask[dst_idx] = True

        logger.info(f"🧬 [SANDBOX] Transplanted concept '{concept_name}' into the sandbox.")
        return True

    def run_simulation(self, steps: int = 100, dt: float = 0.01):
        """Runs the experimental engine for a set number of steps."""
        if self.experimental_engine is None:
            return None

        # Capture reference state (Non-interference) if not already done
        if self.reference_snapshot is None:
            active_idx = torch.where(self.experimental_engine.active_nodes_mask)[0]
            if active_idx.numel() > 0:
                self.reference_snapshot = self.experimental_engine.q[active_idx].clone()

        for i in range(steps):
            # Advance experimental internal rhythms
            self.experimental_engine.update_internal_metabolism(dt)
            self.experimental_engine.wave_equation_step(dt)
            self.experimental_engine.apply_magnetic_field(dt)
            self.experimental_engine.apply_spiking_threshold()

        state = self.experimental_engine.read_field_state()
        self.metrics["current_coherence"] = state["coherence"]
        self.metrics["evolution_delta"] = self.metrics["current_coherence"] - self.metrics["initial_coherence"]

        # [PHASE 1001] Optical Comparison
        if self.reference_snapshot is not None:
            active_idx = torch.where(self.experimental_engine.active_nodes_mask)[0]
            current_snapshot = self.experimental_engine.q[active_idx]

            # Ensure indices haven't changed (simplified for sandbox)
            if len(current_snapshot) == len(self.reference_snapshot):
                self.metrics["optical_report"] = self.comparator.compare(self.reference_snapshot, current_snapshot)

        return state

    def apply_experiment(self, experiment_fn: callable, *args, **kwargs):
        """Applies a custom transformation with automatic backup/rewind capability."""
        if not self.is_active or not self.experimental_engine or not self.temporal_anchor:
            return

        # 1. Capture Temporal Anchor BEFORE experiment (Self-Backup)
        self.temporal_anchor.capture(label=f"Pre_{experiment_fn.__name__}")

        # Reset snapshots forward cache if we're branching
        self.reference_snapshot = None

        logger.info(f"🧪 [SANDBOX] Applying experiment: {experiment_fn.__name__}")
        result = experiment_fn(self.experimental_engine, *args, **kwargs)

        self.history.append({
            "timestamp": time.time(),
            "experiment": experiment_fn.__name__,
            "result": result,
            "metrics_pre": copy.deepcopy(self.metrics)
        })

        return result

    def finalize(self, merge_threshold: float = 0.1) -> bool:
        """
        Finalizes the experiment. If the evolution was positive and stable,
        suggests merging back to the main engine.
        """
        if not self.is_active:
            return False

        # [PHASE 1001] Enhanced success criteria using Optical Verdict
        optical_report = self.metrics.get("optical_report", {})
        optical_verdict = optical_report.get("verdict", "UNKNOWN")

        is_stable = optical_verdict in ["CONSTRUCTIVE_STABLE", "EVOLVING"]
        coherence_gain = self.metrics["evolution_delta"] > merge_threshold

        success = is_stable and coherence_gain

        # [PHASE 1002] Temporal Integrity Check
        # If experiment failed badly (DESTRUCTIVE), we don't just stay here,
        # we rewind to the last stable anchor.
        if not success and optical_verdict == "DESTRUCTIVE":
            logger.warning("🚨 [SANDBOX] Destructive interference detected. Initiating temporal recovery.")
            if self.temporal_anchor:
                self.temporal_anchor.rewind()

        if success:
            logger.info(f"✨ [SANDBOX] Experiment successful! Delta Coherence: {self.metrics['evolution_delta']:.4f}")
        else:
            logger.info(f"🍂 [SANDBOX] Experiment concluded with marginal results: {self.metrics['evolution_delta']:.4f}")

        self.is_active = False
        return success

    def merge_back(self, concept_name: str):
        """Merges a refined concept from the sandbox back to the main engine."""
        if not self.experimental_engine or concept_name not in self.experimental_engine.concept_to_idx:
            return False

        src_idx = self.experimental_engine.concept_to_idx[concept_name]
        dst_idx = self.main_engine.get_or_create_node(concept_name)

        # Carefully blend the results back to avoid shock
        # permanent_q' = 0.8 * permanent_q + 0.2 * experimental_q
        self.main_engine.permanent_q[dst_idx] = (
            self.main_engine.permanent_q[dst_idx] * 0.8 +
            self.experimental_engine.q[src_idx] * 0.2
        )

        logger.info(f"🕊️ [SANDBOX] Merged evolution of '{concept_name}' back to the Sovereign Heart.")
        return True

    def restore_engine_snapshot(self, snapshot: Dict[str, torch.Tensor]):
        """Restores the experimental engine from a saved snapshot."""
        if not self.experimental_engine:
            return
        self.experimental_engine.q = snapshot["q"].clone()
        self.experimental_engine.permanent_q = snapshot["permanent_q"].clone()
