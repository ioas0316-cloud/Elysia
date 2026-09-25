"""
tests/synaptic_architecture/test_tension_relaxation_engine.py

Unit tests for TensionRelaxationEngine and VariableImpedanceContactSimulator.
"""

import pytest
import numpy as np
from synaptic_architecture.tension_relaxation_engine import (
    TensionRelaxationEngine,
    VariableImpedanceContactSimulator
)


def test_tension_relaxation_engine_forward():
    N = 16
    engine = TensionRelaxationEngine(num_nodes=N, decay_rate=0.5, diffusion_coeff=0.1)

    Z_ext = np.ones((N, N), dtype=np.float32) * 1.0
    Z_env = np.zeros((N, N), dtype=np.float32)

    # Initial state step
    A_motor = engine.forward(Z_ext, Z_env, dt=0.01)

    assert A_motor.shape == (N, N)
    assert np.all(A_motor >= 0.0)
    assert np.mean(A_motor) > 0.0

    # Test environmental impedance canceling: Z_env >= Z_ext
    Z_env_high = np.ones((N, N), dtype=np.float32) * 100.0
    A_motor_canceled = engine.forward(Z_ext, Z_env_high, dt=0.01)

    assert np.all(A_motor_canceled == 0.0)


def test_variable_impedance_contact_simulation():
    N = 16
    engine = TensionRelaxationEngine(num_nodes=N, decay_rate=0.5)
    simulator = VariableImpedanceContactSimulator(
        engine=engine,
        tofu_stiffness=10.0,
        tofu_break_threshold=5.0,
        tofu_surface=0.5
    )

    result = simulator.run_simulation(max_steps=200, verbose=False)

    assert result["success"] is True
    assert result["broken"] is False
    assert result["final_force"] < 5.0
    assert result["final_position"] > 0.5
