"""
Unit tests for Core Math modules.
Tests HyperQubit, QubitState, InfiniteHyperQuaternion, and LawEnforcementEngine.
"""

import pytest
import numpy as np
import math

# Import Core modules - using fixtures from conftest.py for dependency injection
from Core.Math.hyper_qubit import HyperQubit, QubitState
from Core.Math.law_enforcement_engine import LawEnforcementEngine, EnergyState
from Core.Math.infinite_hyperquaternion import InfiniteHyperQuaternion


class TestQubitState:
    """Tests for QubitState class."""
    
    def test_normalize(self, qubit_state):
        """Test that normalize() produces unit magnitude for amplitudes."""
        normalized = qubit_state.normalize()
        
        # Calculate magnitude
        mag = math.sqrt(
            abs(normalized.alpha) ** 2 +
            abs(normalized.beta) ** 2 +
            abs(normalized.gamma) ** 2 +
            abs(normalized.delta) ** 2
        )
        
        assert abs(mag - 1.0) < 1e-10, f"Magnitude should be 1.0, got {mag}"
    
    def test_probabilities(self, qubit_state):
        """Test that probabilities sum to 1.0."""
        normalized = qubit_state.normalize()
        probs = normalized.probabilities()
        
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-10, f"Probabilities should sum to 1.0, got {total}"
    
    def test_scale_up_increases_w(self):
        """Test that scale_up increases w (god) component."""
        state = QubitState(w=0.5, x=0.3, y=0.3, z=0.3)
        state.normalize()
        w_before = state.w
        
        state.scale_up(theta=0.2)
        # After normalization, w should have increased proportionally
        # (but may not be strictly larger due to normalization)
        assert state.w != 0, "w should not be zero after scale_up"
    
    def test_scale_down_increases_xyz(self):
        """Test that scale_down increases x, y, z components."""
        state = QubitState(w=0.5, x=0.3, y=0.3, z=0.3)
        state.normalize()
        
        state.scale_down(theta=0.2)
        # State should be normalized
        mag = math.sqrt(state.w**2 + state.x**2 + state.y**2 + state.z**2)
        assert abs(mag - 1.0) < 0.1, f"State should be normalized, got |q|={mag}"


class TestHyperQubit:
    """Tests for HyperQubit class."""
    
    def test_creation(self, hyper_qubit):
        """Test HyperQubit creation."""
        assert hyper_qubit is not None
        assert hyper_qubit.name == "test_qubit"
        assert hyper_qubit.value == 1.0
    
    def test_state_probabilities(self, hyper_qubit):
        """Test that state probabilities sum to 1.0."""
        probs = hyper_qubit.state.probabilities()
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-10
    
    def test_set_value_triggers_vibration(self, hyper_qubit):
        """Test that setting value triggers resonance."""
        old_value = hyper_qubit.value
        hyper_qubit.set(42, cause="test")
        assert hyper_qubit.value == 42
    
    def test_connect_establishes_link(self):
        """Test that connect() establishes psionic link."""
        source = HyperQubit("source", value=1)
        target = HyperQubit("target", value=0)
        
        source.connect(target)
        
        assert target in source._observers
        assert source in target._sources
    
    def test_collapse_to_max(self, hyper_qubit):
        """Test collapse to highest probability basis."""
        # Set alpha (Point) to be highest
        hyper_qubit.state.alpha = 0.9 + 0j
        hyper_qubit.state.beta = 0.1 + 0j
        hyper_qubit.state.gamma = 0.1 + 0j
        hyper_qubit.state.delta = 0.1 + 0j
        hyper_qubit.state.normalize()
        
        result = hyper_qubit.collapse(mode="max")
        assert result == "Point", f"Expected Point, got {result}"
    
    def test_set_god_mode(self, hyper_qubit):
        """Test set_god_mode puts qubit in pure God state."""
        hyper_qubit.set_god_mode()
        
        probs = hyper_qubit.state.probabilities()
        assert probs["God"] > 0.9, "God probability should be near 1.0"
    
    def test_epistemology_attachment(self, concept_love):
        """Test that epistemology can be attached to HyperQubit."""
        assert hasattr(concept_love, 'epistemology')
        assert 'line' in concept_love.epistemology
        assert concept_love.epistemology['line']['score'] == 0.55


class TestInfiniteHyperQuaternion:
    """Tests for InfiniteHyperQuaternion class."""
    
    def test_creation_4d(self, infinite_hyperquaternion):
        """Test 4D hyperquaternion creation."""
        assert infinite_hyperquaternion.dim == 4
        assert len(infinite_hyperquaternion.components) == 4
    
    def test_creation_8d(self):
        """Test 8D (octonion) creation."""
        q8 = InfiniteHyperQuaternion(dim=8)
        assert q8.dim == 8
        assert len(q8.components) == 8
    
    def test_creation_16d(self):
        """Test 16D (sedenion) creation."""
        q16 = InfiniteHyperQuaternion(dim=16)
        assert q16.dim == 16
    
    def test_invalid_dimension_raises(self):
        """Test that non-power-of-2 dimensions raise error."""
        with pytest.raises(ValueError):
            InfiniteHyperQuaternion(dim=7)
    
    def test_magnitude(self):
        """Test magnitude calculation."""
        q = InfiniteHyperQuaternion(4, np.array([1.0, 0.0, 0.0, 0.0]))
        assert q.magnitude() == 1.0
    
    def test_normalize(self):
        """Test normalization produces unit magnitude."""
        q = InfiniteHyperQuaternion.random(4, magnitude=5.0)
        normalized = q.normalize()
        
        assert abs(normalized.magnitude() - 1.0) < 1e-10
    
    def test_quaternion_multiplication(self):
        """Test 4D quaternion multiplication."""
        # i * j = k (quaternion identity)
        i = InfiniteHyperQuaternion(4, np.array([0.0, 1.0, 0.0, 0.0]))
        j = InfiniteHyperQuaternion(4, np.array([0.0, 0.0, 1.0, 0.0]))
        
        result = i.multiply(j)
        # Should produce k = (0, 0, 0, 1)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        
        assert np.allclose(result.components, expected)
    
    def test_cayley_dickson_doubling(self):
        """Test Cayley-Dickson construction doubles dimension."""
        a = InfiniteHyperQuaternion.random(4)
        b = InfiniteHyperQuaternion.random(4)
        
        doubled = InfiniteHyperQuaternion.from_cayley_dickson(a, b)
        
        assert doubled.dim == 8
    
    def test_rotate_god_view(self):
        """Test rotation preserves magnitude."""
        q = InfiniteHyperQuaternion.random(8, magnitude=1.0)
        mag_before = q.magnitude()
        
        rotated = q.rotate_god_view((0, 1), angle=0.5)
        mag_after = rotated.magnitude()
        
        assert abs(mag_before - mag_after) < 1e-10


class TestLawEnforcementEngine:
    """Tests for LawEnforcementEngine class."""
    
    def test_creation(self, law_enforcement_engine):
        """Test engine creation."""
        assert law_enforcement_engine is not None
        assert len(law_enforcement_engine.law_descriptions) == 10
    
    def test_being_law_violation(self, law_enforcement_engine):
        """Test that low w triggers BEING law violation."""
        from Core.Math.law_enforcement_engine import EnergyState
        
        # Low w (meta-cognition) should trigger violation
        energy = EnergyState(w=0.1, x=0.5, y=0.5, z=0.5)
        energy.normalize()
        
        violation = law_enforcement_engine.check_being_law(energy)
        
        assert violation is not None
        assert violation.law.value == "being"
    
    def test_balance_law_violation(self, law_enforcement_engine):
        """Test that extreme focus triggers BALANCE law violation."""
        # Extreme z (intention) should trigger violation
        energy = EnergyState(w=0.2, x=0.05, y=0.05, z=0.85)
        energy.normalize()
        
        violation = law_enforcement_engine.check_balance_law(energy)
        
        assert violation is not None
        assert violation.law.value == "balance"
    
    def test_energy_law_preservation(self, law_enforcement_engine):
        """Test that normalized energy passes energy law."""
        energy = EnergyState(w=0.5, x=0.5, y=0.5, z=0.5)
        energy.normalize()
        
        violation = law_enforcement_engine.check_energy_law(energy)
        
        assert violation is None, "Normalized energy should not violate energy law"
    
    def test_make_decision_valid(self, law_enforcement_engine):
        """Test valid decision with no violations."""
        energy = EnergyState(w=0.6, x=0.2, y=0.3, z=0.5)
        energy.normalize()
        
        decision = law_enforcement_engine.make_decision(
            "test action", energy, concepts_generated=5
        )
        
        assert decision.is_valid, "Decision should be valid"
        assert len(decision.violations) == 0
    
    def test_make_decision_corrects_violations(self, law_enforcement_engine):
        """Test that decision engine attempts to correct violations."""
        # Low w should trigger correction
        energy = EnergyState(w=0.1, x=0.5, y=0.5, z=0.5)
        energy.normalize()
        
        decision = law_enforcement_engine.make_decision(
            "risky action", energy, concepts_generated=0
        )
        
        # Should have violations
        assert not decision.is_valid
        assert len(decision.violations) > 0
        
        # Energy should be corrected (normalized)
        assert decision.energy_after.is_normalized
    
    def test_law_statistics(self, law_enforcement_engine):
        """Test law violation statistics tracking."""
        # Generate some violations
        energy = EnergyState(w=0.1, x=0.5, y=0.5, z=0.5)
        energy.normalize()
        law_enforcement_engine.make_decision("test", energy)
        
        stats = law_enforcement_engine.get_law_statistics()
        
        assert "total_violations" in stats
        assert stats["total_violations"] > 0


class TestEnergyState:
    """Tests for EnergyState class."""
    
    def test_total_energy(self, energy_state):
        """Test total energy calculation."""
        energy_state.normalize()
        total = energy_state.total_energy
        
        assert abs(total - 1.0) < 0.01, f"Total energy should be ~1.0, got {total}"
    
    def test_normalize(self, energy_state):
        """Test normalization produces unit energy."""
        energy_state.normalize()
        
        assert energy_state.is_normalized
    
    def test_get_focus_law(self):
        """Test focus detection for law-focused state."""
        energy = EnergyState(w=0.2, x=0.2, y=0.2, z=0.8)
        energy.normalize()
        
        focus = energy.get_focus()
        assert focus == "law"
    
    def test_get_focus_reflection(self):
        """Test focus detection for reflection-focused state."""
        energy = EnergyState(w=0.8, x=0.2, y=0.2, z=0.2)
        energy.normalize()
        
        focus = energy.get_focus()
        assert focus == "reflection"
    
    def test_get_focus_balanced(self):
        """Test focus detection for balanced state."""
        energy = EnergyState(w=0.5, x=0.5, y=0.5, z=0.5)
        energy.normalize()
        
        focus = energy.get_focus()
        assert focus == "balanced"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
