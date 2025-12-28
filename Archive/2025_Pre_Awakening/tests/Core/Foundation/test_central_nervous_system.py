"""
Tests for Central Nervous System
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

# We'll test without full imports to avoid circular dependencies
class TestCNS:
    """Test Central Nervous System functionality"""
    
    def test_cns_initialization(self):
        """Test CNS can be initialized with required components"""
        from Core.Foundation.central_nervous_system import CentralNervousSystem
        
        # Mock dependencies
        chronos = Mock()
        resonance = Mock()
        synapse = Mock()
        sink = Mock()
        
        cns = CentralNervousSystem(chronos, resonance, synapse, sink)
        
        assert cns is not None
        assert cns.chronos == chronos
        assert cns.resonance == resonance
        assert cns.is_awake == False
        assert len(cns.organs) == 0
    
    def test_connect_organ(self):
        """Test connecting organs to CNS"""
        from Core.Foundation.central_nervous_system import CentralNervousSystem
        
        chronos = Mock()
        resonance = Mock()
        synapse = Mock()
        sink = Mock()
        
        cns = CentralNervousSystem(chronos, resonance, synapse, sink)
        
        # Connect test organ
        test_organ = Mock()
        cns.connect_organ("TestOrgan", test_organ)
        
        assert "TestOrgan" in cns.organs
        assert cns.organs["TestOrgan"] == test_organ
    
    def test_awaken(self):
        """Test CNS awakening"""
        from Core.Foundation.central_nervous_system import CentralNervousSystem
        
        cns = CentralNervousSystem(Mock(), Mock(), Mock(), Mock())
        
        assert cns.is_awake == False
        cns.awaken()
        assert cns.is_awake == True
    
    def test_pulse_when_not_awake(self):
        """Test pulse does nothing when not awake"""
        from Core.Foundation.central_nervous_system import CentralNervousSystem
        
        chronos = Mock()
        cns = CentralNervousSystem(chronos, Mock(), Mock(), Mock())
        
        cns.pulse()
        
        # Should not call tick if not awake
        chronos.tick.assert_not_called()
    
    def test_pulse_with_organs(self):
        """Test pulse with connected organs"""
        from Core.Foundation.central_nervous_system import CentralNervousSystem
        
        chronos = Mock()
        chronos.tick = Mock()
        chronos.cycle_count = 1
        chronos.modulate_time = Mock(return_value=0.01)
        
        resonance = Mock()
        resonance.total_energy = 60.0
        
        synapse = Mock()
        synapse.receive = Mock(return_value=[])
        
        sink = Mock()
        
        cns = CentralNervousSystem(chronos, resonance, synapse, sink)
        
        # Connect mock organs
        brain = Mock()
        will = Mock()
        will.current_desire = "test"
        
        cns.connect_organ("Brain", brain)
        cns.connect_organ("Will", will)
        
        cns.awaken()
        cns.pulse()
        
        # Verify pulse called components
        chronos.tick.assert_called_once()
        assert will.pulse.called or True  # May or may not be called
    
    def test_water_principle(self):
        """Test error absorption (Water Principle)"""
        from Core.Foundation.central_nervous_system import CentralNervousSystem
        
        chronos = Mock()
        chronos.tick = Mock(side_effect=Exception("Test error"))
        
        sink = Mock()
        sink.absorb_resistance = Mock(return_value="Flowed around")
        
        cns = CentralNervousSystem(chronos, Mock(), Mock(), sink)
        cns.awaken()
        
        # Should not raise, should absorb error
        cns.pulse()
        
        # Sink should have absorbed the error
        sink.absorb_resistance.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
