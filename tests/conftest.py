"""
Pytest configuration and fixtures for Elysia tests.
Provides reusable test fixtures and configuration.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def hyper_qubit():
    """Create a test HyperQubit instance."""
    from Core.Foundation.Math.hyper_qubit import HyperQubit
    return HyperQubit("test_qubit", value=1.0, name="test_qubit")


@pytest.fixture
def qubit_state():
    """Create a test QubitState instance."""
    from Core.Foundation.Math.hyper_qubit import QubitState
    return QubitState(
        alpha=0.5 + 0j,
        beta=0.5 + 0j,
        gamma=0.5 + 0j,
        delta=0.5 + 0j,
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0
    )


@pytest.fixture
def law_enforcement_engine():
    """Create a test LawEnforcementEngine instance."""
    from Core.Foundation.Math.law_enforcement_engine import LawEnforcementEngine
    return LawEnforcementEngine()


@pytest.fixture
def energy_state():
    """Create a test EnergyState instance."""
    from Core.Foundation.Math.law_enforcement_engine import EnergyState
    return EnergyState(w=0.6, x=0.2, y=0.3, z=0.5)


@pytest.fixture
def infinite_hyperquaternion():
    """Create a test InfiniteHyperQuaternion instance."""
    from Core.Foundation.Math.infinite_hyperquaternion import InfiniteHyperQuaternion
    return InfiniteHyperQuaternion(dim=4)


@pytest.fixture
def concept_love():
    """Create a 'love' concept HyperQubit with epistemology."""
    from Core.Foundation.Math.hyper_qubit import HyperQubit, QubitState
    
    qubit = HyperQubit(
        "love",
        value="Universal binding force",
        name="love"
    )
    
    # Set epistemology for philosophical meaning
    qubit.epistemology = {
        "point": {"score": 0.15, "meaning": "neurochemistry is substrate only"},
        "line": {"score": 0.55, "meaning": "Spinoza's universal binding"},
        "space": {"score": 0.20, "meaning": "field effect, mutual resonance"},
        "god": {"score": 0.10, "meaning": "transcendent purpose (Heidegger)"}
    }
    
    # Set state to match epistemology
    state = QubitState(
        alpha=0.15 + 0j,  # Point
        beta=0.55 + 0j,   # Line
        gamma=0.20 + 0j,  # Space
        delta=0.10 + 0j   # God
    )
    qubit.state = state.normalize()
    
    return qubit


@pytest.fixture
def concept_truth():
    """Create a 'truth' concept HyperQubit with epistemology."""
    from Core.Foundation.Math.hyper_qubit import HyperQubit, QubitState
    
    qubit = HyperQubit(
        "truth",
        value="Correspondence with reality",
        name="truth"
    )
    
    qubit.epistemology = {
        "point": {"score": 0.30, "meaning": "empirical verification"},
        "line": {"score": 0.25, "meaning": "logical coherence"},
        "space": {"score": 0.35, "meaning": "pragmatic usefulness"},
        "god": {"score": 0.10, "meaning": "absolute/transcendent truth"}
    }
    
    state = QubitState(
        alpha=0.30 + 0j,
        beta=0.25 + 0j,
        gamma=0.35 + 0j,
        delta=0.10 + 0j
    )
    qubit.state = state.normalize()
    
    return qubit
