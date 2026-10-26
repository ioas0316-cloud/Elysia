"""
test_particle_rotor_matrix_sandbox.py: Tests for Micro-Rotor Particle Matrix Sandbox and Timeline Scrubber.
"""

import pytest
import numpy as np
from L1_causal_field.core.topology.particle_rotor_matrix_sandbox import (
    MicroRotorParticle,
    ParticleRotorMatrixSandbox,
    CausalTimelineScrubber,
)


def test_micro_rotor_particle_initialization():
    p = MicroRotorParticle(
        id=1,
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.1, -0.1]),
        mass=2.0,
        friction=0.2,
        theta=np.pi / 4,
        omega=1.5,
    )
    assert p.id == 1
    assert p.mass == 2.0
    assert "pressure" in p.sensory_profile
    assert p.sensory_profile["weight"] == 2.0


def test_sandbox_emergence_and_step():
    sandbox = ParticleRotorMatrixSandbox(dimensions=2, bounds=10.0, dt=0.05)
    for i in range(10):
        pos = np.random.uniform(-5.0, 5.0, size=2)
        vel = np.random.uniform(-0.5, 0.5, size=2)
        theta = np.random.uniform(0, 2 * np.pi)
        p = MicroRotorParticle(id=i, position=pos, velocity=vel, theta=theta)
        sandbox.add_particle(p)

    initial_order = sandbox.get_macro_order_parameter()
    assert 0.0 <= initial_order <= 1.0

    step_info = sandbox.step()
    assert step_info["step_idx"] == 1
    assert "macro_order" in step_info
    assert "total_energy" in step_info


def test_timeline_scrubbing_and_counterfactual():
    sandbox = ParticleRotorMatrixSandbox(dimensions=2, bounds=10.0, dt=0.05)
    for i in range(5):
        p = MicroRotorParticle(
            id=i,
            position=np.array([float(i), float(i)]),
            velocity=np.array([0.1, 0.0]),
            theta=0.0,
        )
        sandbox.add_particle(p)

    # Run for 20 steps
    for _ in range(20):
        sandbox.step()

    assert len(sandbox.scrubber.history) == 20

    # Scrub back to step 10
    frame_10 = sandbox.scrubber.scrub_to(10)
    assert frame_10 is not None
    assert frame_10["step_idx"] == 10

    # Perform counterfactual trace
    def perturb(particles):
        for p in particles:
            p.theta = np.pi # Reverse phase

    def sandbox_factory(particles):
        return ParticleRotorMatrixSandbox(particles=particles, dimensions=2, bounds=10.0, dt=0.05)

    cf_result = sandbox.scrubber.counterfactual_trace(
        start_step=10,
        steps_forward=5,
        perturbation_fn=perturb,
        sandbox_factory=sandbox_factory,
    )

    assert cf_result["start_step"] == 10
    assert len(cf_result["forked_order_history"]) == 5
    assert cf_result["order_divergence"] >= 0.0
