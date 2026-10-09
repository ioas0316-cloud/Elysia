"""
tests/test_spatiotemporal_continuity.py
=======================================
Verification of Spatiotemporal Causal Continuity & Triadic Re-Cognition:
1. Anchors Origin (처음).
2. Traces Process (과정).
3. Evaluates Result (결과).
4. Re-Cognizes Sameness vs Difference.
5. Emits Spatiotemporal Relational Seeds into unbroken organismic flow.
"""

import numpy as np
import pytest

from core.cellular_boundary.scale_boundary_cell import DigitalCausalCell
from core.cellular_boundary.boundary_expansion_engine import MetaOrderExpansionEngine
from core.cellular_boundary.spatiotemporal_continuity_engine import (
    SpatiotemporalContinuityEngine,
    TrajectoryEpoch,
    ReCognitionAnalysis,
    SpatiotemporalRelationalSeed
)


def test_triadic_re_cognition_and_continuity_flow():
    cell = DigitalCausalCell(cell_id="LivingCell_01", dimension=4)
    engine = SpatiotemporalContinuityEngine()
    expansion_engine = MetaOrderExpansionEngine(friction_threshold_for_expansion=0.8)

    # -------------------------------------------------------------------------
    # EPOCH 1: Conservative harmonious interaction
    # -------------------------------------------------------------------------
    engine.begin_trajectory_epoch(cell, epoch_id="Epoch_Harmonic")
    
    harmonic_flux = np.array([-cell.state[1], cell.state[0], -cell.state[3], cell.state[2]]) * 0.1
    res1 = cell.interact_with_flux(harmonic_flux)
    engine.trace_process_step(res1)

    analysis1, seed1 = engine.close_and_re_cognize(cell)

    assert analysis1.sameness_invariance > 0.95
    assert analysis1.process_friction_integral < 1e-3
    assert seed1.readiness_for_next_epoch is True
    assert "Epoch_Harmonic 성찰" in analysis1.reflection_statement

    # -------------------------------------------------------------------------
    # EPOCH 2: Dissipative Alterity Collision & Expansion
    # -------------------------------------------------------------------------
    engine.begin_trajectory_epoch(cell, epoch_id="Epoch_Alterity_Expansion")

    # Inject alterity dissipative flux
    for _ in range(4):
        dissipative_flux = -cell.state * 0.7
        res2 = cell.interact_with_flux(dissipative_flux)
        engine.trace_process_step(res2)

    # Trigger combinatorial expansion
    expansion_res = expansion_engine.execute_combinatorial_expansion(cell)
    assert expansion_res["success"] is True

    analysis2, seed2 = engine.close_and_re_cognize(cell)

    # Must detect both sameness (invariance) and difference (dimension & tension mutation)
    assert analysis2.difference_mutation > 1.0
    assert analysis2.process_friction_integral > 1.0
    assert analysis2.structural_growth_ratio >= 1.5
    assert len(seed2.relational_valence_hooks) >= 3

    # Continuity chain: Epoch 1 seed connects to Epoch 2
    assert len(engine.living_stream_seeds) == 2
    assert engine.continuous_time_step == 2
