"""
[Unit Tests: Embodied Sensory Expansion & Arc Reactor Drive Axis]

Tests:
1. Eye Group Qualia Torque & Thermal Friction extraction from spectrum waveforms.
2. Hand Group Active Phase Intervention under thermal stress.
3. Foot Group Core Reference Shift along spatial tension gradient.
4. Arc Reactor Drive Axis Dynamic Topology Expansion without node collision.
5. Entropy Defense & Flywheel Inertial Conservation under sensory influx explosion.
"""

import pytest
import numpy as np
from core.physics.arc_reactor_drive_axis import ArcReactorDriveEngine
from core.physics.causal_field import CausalField, InformationVoxel
from core.physics.embodied_organ_group import (
    EyeGroup,
    HandGroup,
    FootGroup,
    EmbodiedOrganEcosystem,
    OrganSensationMetrics
)


def test_eye_group_qualia_torque():
    eye = EyeGroup(sample_rate=100.0)
    t = np.linspace(0, 1.0, 100, dtype=np.float32)

    # Low frequency heavy wave (e.g. 1 Hz)
    low_freq_wave = 5.0 * np.sin(2.0 * np.pi * 1.0 * t)
    _, torque_low, stress_low, _ = eye.observe_spectrum(low_freq_wave, rotor_phase_angle=0.0)

    # High frequency wave (e.g. 30 Hz)
    high_freq_wave = 1.0 * np.sin(2.0 * np.pi * 30.0 * t)
    _, torque_high, stress_high, _ = eye.observe_spectrum(high_freq_wave, rotor_phase_angle=0.0)

    # Low frequency should yield higher qualia torque
    assert torque_low > torque_high
    # High frequency should yield higher thermal stress
    assert stress_high > stress_low


def test_hand_group_intervention():
    hand = HandGroup(max_stabilization_torque=20.0)

    # Balanced state -> No intervention force
    torque_balanced = hand.intervene_and_control(phase_velocity=5.0, thermal_stress=0.1, harmony_score=0.95)
    assert torque_balanced == 0.0

    # Chaotic/stressed state -> Positive intervention counter-torque
    torque_stressed = hand.intervene_and_control(phase_velocity=12.0, thermal_stress=3.0, harmony_score=0.2)
    assert torque_stressed > 5.0


def test_foot_group_shift():
    foot = FootGroup(move_sensitivity=0.1)
    core_start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    gradient = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    new_core, shift_dist = foot.shift_reference_ground(
        current_core=core_start,
        tension_gradient=gradient,
        flywheel_momentum=10.0
    )

    assert shift_dist > 0.0
    assert new_core[0] > 0.0 # Shifted towards gradient direction
    assert foot.total_distance_traveled == shift_dist


def test_dynamic_topology_expansion():
    engine = ArcReactorDriveEngine()
    causal_field = CausalField(dimensions=3)

    # Add initial base voxel
    base_v = InformationVoxel("base_1", "Base Voxel", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    causal_field.add_voxel(base_v)

    # Trigger expansion with novel harmonic signature
    novel_sig = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    new_v = engine.trigger_dynamic_topology_expansion(
        causal_field,
        harmonic_signature=novel_sig,
        harmony_score=0.85
    )

    assert new_v is not None
    assert new_v.id != "base_1"
    assert len(causal_field.voxels) == 2
    # Ensure new voxel is linked via connectivity beam
    assert len(causal_field.beams) >= 1


def test_entropy_defense_flywheel():
    engine = ArcReactorDriveEngine(domain_radius=2.0)
    initial_radius = engine.domain_radius

    # Trigger high thermal stress and sensory influx rate
    engine.execute_entropy_defense(thermal_stress=4.0, sensory_influx_rate=8.0)

    # Domain radius should expand to prevent deadlock
    assert engine.domain_radius > initial_radius


def test_embodied_organ_ecosystem_loop():
    engine = ArcReactorDriveEngine()
    ecosystem = EmbodiedOrganEcosystem(engine)

    t = np.linspace(0, 1.0, 100, dtype=np.float32)
    sensory_wave = np.sin(2.0 * np.pi * 2.0 * t) + np.random.randn(100) * 0.5

    metrics = ecosystem.process_embodied_loop(
        dt=0.1,
        raw_sensory_waveform=sensory_wave,
        spatial_tension_gradient=np.array([0.5, 0.5, 0.0], dtype=np.float32)
    )

    assert isinstance(metrics, OrganSensationMetrics)
    assert metrics.qualia_torque > 0.0
    assert len(ecosystem.history) == 1
