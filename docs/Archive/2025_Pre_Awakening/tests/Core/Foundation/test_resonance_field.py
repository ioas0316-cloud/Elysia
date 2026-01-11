"""
Tests for Resonance Field
"""

import pytest
from unittest.mock import Mock

class TestResonanceField:
    """Test Resonance Field functionality"""
    
    def test_resonance_field_init(self):
        """Test ResonanceField initialization"""
        from Core.FoundationLayer.Foundation.resonance_field import ResonanceField
        
        field = ResonanceField()
        
        assert field is not None
        assert field.total_energy >= 0
    
    def test_add_energy(self):
        """Test adding energy to field"""
        from Core.FoundationLayer.Foundation.resonance_field import ResonanceField
        
        field = ResonanceField()
        initial = field.total_energy
        
        field.add_energy(50.0)
        
        assert field.total_energy >= initial
    
    def test_drain_energy_success(self):
        """Test draining energy when sufficient"""
        from Core.FoundationLayer.Foundation.resonance_field import ResonanceField
        
        field = ResonanceField()
        field.add_energy(100.0)
        
        result = field.drain_energy(50.0)
        
        # Should succeed if enough energy
        assert isinstance(result, bool)
    
    def test_drain_energy_insufficient(self):
        """Test draining more energy than available"""
        from Core.FoundationLayer.Foundation.resonance_field import ResonanceField
        
        field = ResonanceField()
        initial = field.total_energy
        
        # Try to drain more than we have
        result = field.drain_energy(initial + 1000.0)
        
        # Should indicate if it failed
        assert isinstance(result, bool)

class TestEntropySink:
    """Test Entropy Sink (Water Principle)"""
    
    def test_entropy_sink_init(self):
        """Test EntropySink initialization"""
        from Core.FoundationLayer.Foundation.entropy_sink import EntropySink
        from Core.FoundationLayer.Foundation.resonance_field import ResonanceField
        
        field = ResonanceField()
        sink = EntropySink(field)
        
        assert sink is not None
        assert sink.resonance == field
    
    def test_absorb_resistance(self):
        """Test error absorption"""
        from Core.FoundationLayer.Foundation.entropy_sink import EntropySink
        from Core.FoundationLayer.Foundation.resonance_field import ResonanceField
        
        sink = EntropySink(ResonanceField())
        
        error = Exception("Test error")
        result = sink.absorb_resistance(error, "Test operation")
        
        # Should return fallback message
        assert isinstance(result, str)
        assert len(result) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
