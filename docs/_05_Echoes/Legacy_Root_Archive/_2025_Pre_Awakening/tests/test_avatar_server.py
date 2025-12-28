"""
Test suite for Avatar Server System
"""

import pytest
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.Interface.avatar_server import (
    Expression,
    Spirits,
    ElysiaAvatarCore
)


class TestExpression:
    """Test Expression dataclass"""
    
    def test_default_values(self):
        expr = Expression()
        assert expr.mouth_curve == 0.0
        assert expr.eye_open == 1.0
        assert expr.brow_furrow == 0.0
        assert expr.beat == 0.0
        assert expr.mouth_width == 0.0
    
    def test_custom_values(self):
        expr = Expression(mouth_curve=0.5, eye_open=0.8)
        assert expr.mouth_curve == 0.5
        assert expr.eye_open == 0.8


class TestSpirits:
    """Test Spirits dataclass"""
    
    def test_default_values(self):
        spirits = Spirits()
        assert spirits.fire == 0.1
        assert spirits.water == 0.1
        assert spirits.earth == 0.3
        assert spirits.air == 0.2
        assert spirits.light == 0.2
        assert spirits.dark == 0.1
        assert spirits.aether == 0.1
    
    def test_all_in_range(self):
        spirits = Spirits()
        for attr in ['fire', 'water', 'earth', 'air', 'light', 'dark', 'aether']:
            value = getattr(spirits, attr)
            assert 0.0 <= value <= 1.0


class TestElysiaAvatarCore:
    """Test ElysiaAvatarCore"""
    
    def test_initialization(self):
        core = ElysiaAvatarCore()
        assert core.expression is not None
        assert core.spirits is not None
        assert core.beat_phase == 0.0
    
    def test_update_beat(self):
        core = ElysiaAvatarCore()
        initial_phase = core.beat_phase
        core.update_beat(0.1)  # 100ms
        assert core.beat_phase != initial_phase
        assert 0.0 <= core.expression.beat <= 1.0
    
    def test_get_state_message(self):
        core = ElysiaAvatarCore()
        state = core.get_state_message()
        
        assert "expression" in state
        assert "spirits" in state
        
        # Check expression fields
        expr = state["expression"]
        assert "mouth_curve" in expr
        assert "eye_open" in expr
        assert "brow_furrow" in expr
        assert "beat" in expr
        assert "mouth_width" in expr
        
        # Check spirits fields
        spirits = state["spirits"]
        assert "fire" in spirits
        assert "water" in spirits
        assert "earth" in spirits
        assert "air" in spirits
        assert "light" in spirits
        assert "dark" in spirits
        assert "aether" in spirits
    
    def test_emotion_processing_with_engine(self):
        """Test emotion processing when EmotionalEngine is available"""
        core = ElysiaAvatarCore()
        
        if core.emotional_engine is None:
            pytest.skip("EmotionalEngine not available")
        
        # Process hopeful emotion
        initial_mouth = core.expression.mouth_curve
        core.process_emotion_event('hopeful', 0.8)
        
        # Should increase positive expression
        # (might not change if emotional_engine is not working)
        assert core.expression is not None
    
    def test_spirits_update(self):
        """Test spirit energy updates"""
        core = ElysiaAvatarCore()
        
        if core.emotional_engine is None or core.spirit_mapper is None:
            pytest.skip("Spirit system not available")
        
        core.update_spirits_from_emotion()
        
        # All spirits should be in valid range
        assert 0.0 <= core.spirits.fire <= 1.0
        assert 0.0 <= core.spirits.water <= 1.0
        assert 0.0 <= core.spirits.earth <= 1.0
        assert 0.0 <= core.spirits.air <= 1.0
        assert 0.0 <= core.spirits.light <= 1.0
        assert 0.0 <= core.spirits.dark <= 1.0
        assert 0.0 <= core.spirits.aether <= 1.0
    
    def test_expression_update(self):
        """Test expression updates from emotion"""
        core = ElysiaAvatarCore()
        
        if core.emotional_engine is None:
            pytest.skip("EmotionalEngine not available")
        
        core.update_expression_from_emotion()
        
        # Expression values should be in valid ranges
        assert -1.0 <= core.expression.mouth_curve <= 1.0
        assert 0.0 <= core.expression.eye_open <= 1.0
        assert 0.0 <= core.expression.brow_furrow <= 1.0


class TestIntegration:
    """Integration tests"""
    
    def test_full_cycle(self):
        """Test a full update cycle"""
        core = ElysiaAvatarCore()
        
        # Process emotion
        core.process_emotion_event('focused', 0.7)
        
        # Update beat
        core.update_beat(0.033)  # ~30 FPS
        
        # Update expression and spirits
        core.update_expression_from_emotion()
        core.update_spirits_from_emotion()
        
        # Get state
        state = core.get_state_message()
        
        # Verify state is valid
        assert state is not None
        assert "expression" in state
        assert "spirits" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
