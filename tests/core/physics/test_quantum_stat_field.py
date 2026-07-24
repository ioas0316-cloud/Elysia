import numpy as np
import pytest
from core.physics.quantum_stat_field import QuantumStatField, StatNode, CatastropheVector

def test_quantum_stat_field_initialization():
    # Test default initialization
    qsf = QuantumStatField()
    assert len(qsf.nodes) == 5
    assert "health" in qsf.nodes
    assert "force" in qsf.nodes
    assert "mind" in qsf.nodes
    assert "speed" in qsf.nodes
    assert "intelligence" in qsf.nodes

    # Check default position properties
    pos_health = qsf.nodes["health"].position
    # Health should be at angle pi/2, so (0, 5, 0)
    assert np.allclose(pos_health, [0.0, 5.0, 0.0], atol=1e-5)

    # Masses should match initial stat value (10.0)
    for node in qsf.nodes.values():
        assert node.mass == 10.0
        assert node.base_value == 10.0

def test_stat_field_evolution_and_convergence():
    qsf = QuantumStatField()

    # Store initial positions
    initial_positions = {name: node.position.copy() for name, node in qsf.nodes.items()}

    # Step the field multiple times
    for _ in range(50):
        qsf.step(dt=0.1)

    # Verify positions have changed (dynamic movement)
    for name, node in qsf.nodes.items():
        assert not np.allclose(node.position, initial_positions[name])
        # Velvet damping should keep coordinates bounded
        assert np.linalg.norm(node.position) < 15.0

def test_quantum_stat_field_dispersion_collapse():
    # Force extremely low health ratio to trigger field dispersion
    qsf = QuantumStatField({
        "health": 0.5,
        "force": 20.0,
        "mind": 20.0,
        "speed": 20.0,
        "intelligence": 20.0
    })

    # Run steps so nodes disperse
    for _ in range(30):
        qsf.step(dt=0.2)

    catastrophe = qsf.get_catastrophe_vector()
    assert catastrophe.is_collapsed
    assert catastrophe.type == "Dispersion"
    assert catastrophe.magnitude > 0.0

def test_quantum_stat_field_overload_collapse():
    # Force an extremely high force stat (over 80%) to trigger Overload
    # Keep health and mind at exactly 10% to avoid dispersion collapse
    qsf = QuantumStatField({
        "health": 10.0,
        "force": 80.0,
        "mind": 10.0,
        "speed": 0.0,
        "intelligence": 0.0
    })

    # Run steps to stretch/squash springs and trigger barrier potential repulsion
    for _ in range(20):
        qsf.step(dt=0.1)

    catastrophe = qsf.get_catastrophe_vector()
    assert catastrophe.is_collapsed
    assert catastrophe.type in ["Overload", "Madness"]
    assert catastrophe.magnitude > 0.0

def test_quantum_stat_field_resonance_sparks():
    # We will balance Speed and Intelligence in Golden Ratio to trigger Spacetime Resonance
    # Phi = 1.618
    # Intelligence = 10, Speed = 16.18
    qsf = QuantumStatField({
        "health": 10.0,
        "force": 10.0,
        "mind": 10.0,
        "speed": 16.18,
        "intelligence": 10.0
    })

    # Set matching spatial distance ratio manually for direct check
    qsf.nodes["speed"].position = np.array([-8.09, 0.0, 0.0])
    qsf.nodes["intelligence"].position = np.array([5.0, 0.0, 0.0])

    resonances = qsf.evaluate_resonance()
    assert len(resonances) > 0
    assert resonances[0]["type"] == "Spacetime_Warp"
    assert "시공간 공명" in resonances[0]["name"]
