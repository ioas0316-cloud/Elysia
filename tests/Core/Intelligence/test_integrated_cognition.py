"""
Tests for Intelligence Systems
"""

import pytest
from unittest.mock import Mock

class TestIntegratedCognition:
    """Test Integrated Cognition System"""
    
    def test_thought_wave_creation(self):
        """Test creating ThoughtWave from text"""
        from Core.Intelligence.integrated_cognition_system import (
            IntegratedCognition
        )
        
        cognition = IntegratedCognition()
        wave = cognition.thought_to_wave("Test thought")
        
        assert wave is not None
        assert wave.content == "Test thought"
        assert wave.frequency > 0
        assert wave.amplitude >= 0
        assert 0 <= wave.phase <= 6.28  # 2π
    
    def test_wave_resonance(self):
        """Test resonance between thought waves"""
        from Core.Intelligence.integrated_cognition_system import (
            IntegratedCognition,
            ThoughtWave
        )
        from Core.Foundation.hyper_quaternion import Quaternion
        
        wave1 = ThoughtWave(
            content="Love",
            frequency=100.0,
            amplitude=1.0,
            phase=0.0,
            wavelength=1.0,
            orientation=Quaternion()
        )
        
        wave2 = ThoughtWave(
            content="Joy",
            frequency=100.0,  # Same frequency
            amplitude=1.0,
            phase=0.0,
            wavelength=1.0,
            orientation=Quaternion()
        )
        
        resonance = wave1.resonate_with(wave2)
        
        assert 0.0 <= resonance <= 1.0
        # Same frequency and phase should resonate well
        assert resonance > 0.5

class TestFractalGoalDecomposer:
    """Test Fractal Goal Decomposition"""
    
    def test_decomposer_init(self):
        """Test FractalGoalDecomposer initialization"""
        from Core.Intelligence.fractal_quaternion_goal_system import (
            FractalGoalDecomposer
        )
        
        decomposer = FractalGoalDecomposer()
        assert decomposer is not None
    
    def test_dimensional_analysis(self):
        """Test analyzing goal from different dimensions"""
        from Core.Intelligence.fractal_quaternion_goal_system import (
            FractalGoalDecomposer,
            Dimension,
            HyperDimensionalLens
        )
        
        lens = HyperDimensionalLens(
            dimension=Dimension.PURPOSE,
            perspective=None,
            clarity=1.0
        )
        
        analysis = lens.analyze("Test goal")
        
        assert isinstance(analysis, str)
        assert len(analysis) > 0
        assert "Test goal" in analysis

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
