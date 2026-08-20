"""
Phase 3.5 Active Falsification & Open-Loop Grounding Simulation Test
====================================================================
Verifies that when the system observes strong unrefined external reality (opposing stimulus/reality):
1. It actively projects focus and detects phase divergence / friction rather than relying on pre-packaged signals.
2. It breaks closed-loop autistic inertia (shatters Gimbal Lock).
3. It dissipates internal tension into external energy.
4. It calibrates its internal phase/state and volitional acceleration to align with external causal reality.
"""

import numpy as np
import pytest
from core.physics.causal_field import CausalField, InformationVoxel
from synaptic_architecture.reflection_engram_engine import (
    ReflectionEngramEngine,
    ActiveMirrorCalibrationPipeline
)

def test_active_observation_and_gimbal_lock_break():
    cf = CausalField(dimensions=3)

    # Initial internal hypothesis (Closed Loop State stuck in [1, 0, 0])
    v_stuck = InformationVoxel(
        id="closed_hypothesis",
        content="Autistic Delusion",
        tensor=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([0.01, 0.0, 0.0], dtype=np.float32) # Refuses to move
    )
    cf.add_voxel(v_stuck)

    # Raw external reality opposes internal hypothesis (e.g., [-1.0, 0.5, 0.0])
    raw_reality = np.array([-1.0, 0.5, 0.0], dtype=np.float32)

    # Actively set focus and observe raw stimulus
    cf.set_intentional_focus(np.array([0.5, 0.5, 0.0], dtype=np.float32))
    obs_res = cf.observe_external_stimulus(raw_reality, target_voxel_id="closed_hypothesis")

    assert obs_res["num_observed"] == 1
    obs_info = obs_res["observations"][0]

    # 1. Phase divergence and friction must be detected
    assert obs_info["phase_divergence"] > 2.0 # High divergence
    assert obs_info["friction_score"] > 1.0

    # 2. Gimbal Lock must be unlocked & energy dissipated
    assert obs_info["gimbal_lock_unlocked"] is True
    assert obs_info["dissipated_energy"] > 0.0
    assert cf.total_dissipated_energy > 0.0

    # 3. Internal tensor calibrated towards external reality
    calibrated_tensor = cf.voxels["closed_hypothesis"].tensor
    # Cosine similarity between calibrated tensor and raw reality should be significantly improved
    dot_prod = np.dot(calibrated_tensor / np.linalg.norm(calibrated_tensor), raw_reality / np.linalg.norm(raw_reality))
    assert dot_prod > 0.0 # Shifted towards reality (positive dot product)


def test_active_mirror_calibration_pipeline_self_correction():
    engine = ReflectionEngramEngine(base_threshold=0.5)
    pipeline = ActiveMirrorCalibrationPipeline(engine, base_threshold=0.5)

    # Internal context stuck in closed loop
    c_context = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    current_vel = np.array([0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Harsh unrefined reality
    raw_reality = np.array([-0.8, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    res = pipeline.process_active_observation(
        C_context=c_context,
        raw_external_reality=raw_reality,
        current_velocity=current_vel
    )

    # 1. Scan triggered and inertia paused (velocity reset to 0)
    assert res["scan_triggered"] is True
    assert np.allclose(res["adjusted_velocity"], 0.0)

    # 2. Dissipation occurred
    assert res["dissipated_energy"] > 0.0
    assert pipeline.total_dissipated_friction > 0.0

    # 3. Volitional acceleration redirected towards reality
    a_vol = np.array(res["a_volition"], dtype=np.float32)
    assert a_vol[0] < 0.0 # Accelerates in negative X direction towards reality

    # 4. Context calibrated
    calibrated_c = np.array(res["calibrated_context"], dtype=np.float32)
    assert calibrated_c[0] < c_context[0] # Shifted towards -0.8

    # 5. Engram recorded
    assert len(engine.engrams) == 1
    engram = engine.engrams[0]
    assert engram.T_grounding > 0.5
