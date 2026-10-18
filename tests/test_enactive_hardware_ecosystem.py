"""Unit tests for Enactive Hardware Ecosystem & Topological Rotor Runtime."""

import pytest
import numpy as np
from core.physics.enactive_hardware_ecosystem_runtime import (
    SpatiotemporalSphericalRotorField,
    TopologicalResonanceCache,
    EnactiveHardwareEcosystemRuntime,
)


def test_spatiotemporal_spherical_rotor_field_conservation():
    """Verifies that Clifford Rotor sandwich operations conserve total wavefield norm/energy."""
    rotor_field = SpatiotemporalSphericalRotorField(num_points=400)
    initial_energy = rotor_field.compute_wavefield_energy()

    # Apply 8 consecutive Clifford Rotor rotations
    for _ in range(8):
        rotor_field.apply_clifford_rotor(angle=np.pi / 4.0)

    final_energy = rotor_field.compute_wavefield_energy()
    assert pytest.approx(initial_energy, abs=1e-5) == final_energy


def test_spatiotemporal_spherical_rotor_field_scale_rotor():
    """Verifies Scale Rotor (dilator) scaling factor calculation."""
    rotor_field = SpatiotemporalSphericalRotorField(num_points=400)

    rotor_field.apply_scale_rotor(scale_factor=2.0)
    assert pytest.approx(2.0, abs=1e-5) == rotor_field.current_scale_factor

    energy = rotor_field.compute_wavefield_energy()
    assert pytest.approx(4.0, abs=1e-4) == energy


def test_topological_resonance_cache_zero_cost_flow():
    """Verifies Zero-Cost Flow caching when Free Energy is below threshold."""
    cache = TopologicalResonanceCache(free_energy_threshold=0.20)
    manifold = np.sin(np.linspace(0, 2 * np.pi, 50))

    # Initial query locks phase
    res1 = cache.query_or_update("scene_01", manifold)
    assert res1["status"] == "INITIAL_PHASE_LOCK"

    # Query with tiny perturbation -> Zero-Cost Flow
    perturbed_manifold = manifold + 0.01
    res2 = cache.query_or_update("scene_01", perturbed_manifold)
    assert res2["status"] == "ZERO_COST_FLOW"
    assert res2["cache_hit"] is True
    assert res2["compute_saved_ratio"] > 0.8


def test_topological_resonance_cache_dynamic_relock():
    """Verifies Dynamic Localized Phase-Re-locking when Free Energy exceeds threshold."""
    cache = TopologicalResonanceCache(free_energy_threshold=0.10)
    manifold = np.sin(np.linspace(0, 2 * np.pi, 50))

    cache.query_or_update("scene_01", manifold)

    # Large perturbation triggers re-locking
    perturbed_manifold = manifold + 0.5
    res2 = cache.query_or_update("scene_01", perturbed_manifold)
    assert res2["status"] == "DYNAMIC_PHASE_RELOCK"
    assert res2["cache_hit"] is False


def test_enactive_hardware_ecosystem_runtime_friction_adaptation():
    """Verifies that the ecosystem runtime adapts to friction across 4 axes without crashing."""
    runtime = EnactiveHardwareEcosystemRuntime(base_freq_hz=60.0)

    # High stress telemetry
    telemetry = {
        "cpu_utilization": 98.0,
        "cpu_queue_length": 10,
        "ram_used_mb": 15000.0,
        "io_latency_ms": 50.0,
        "thermal_celsius": 88.0,
        "thermal_derivative": 2.0,
    }

    step_res = runtime.step_ecosystem(hardware_telemetry=telemetry, dt=0.01)

    assert step_res["status"] == "FRICTION_ADAPTED"
    assert step_res["frictions"]["total_friction"] > 1.0
    # Thermal modulation should lower wave amplitude to maintain thermal equilibrium
    assert step_res["controls"]["wave_amplitude"] < 1.0
    # Backpressure should throttle IO rate
    assert step_res["controls"]["backpressure_rate"] < 1.0
    # Resonance density should remain bounded > 0
    assert 0.0 < step_res["resonance_density"] <= 1.0
