"""
Simple test script for Avatar Server (no pytest required)
Run: python tests/test_avatar_server_simple.py
"""

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


def test_expression_defaults():
    """Test Expression default values"""
    expr = Expression()
    assert expr.mouth_curve == 0.0, "Default mouth_curve should be 0.0"
    assert expr.eye_open == 1.0, "Default eye_open should be 1.0"
    assert expr.brow_furrow == 0.0, "Default brow_furrow should be 0.0"
    print("✅ Expression defaults test passed")


def test_spirits_defaults():
    """Test Spirits default values"""
    spirits = Spirits()
    assert spirits.fire == 0.1, "Default fire should be 0.1"
    assert spirits.water == 0.1, "Default water should be 0.1"
    assert spirits.earth == 0.3, "Default earth should be 0.3"
    
    # Check all are in valid range
    for attr in ['fire', 'water', 'earth', 'air', 'light', 'dark', 'aether']:
        value = getattr(spirits, attr)
        assert 0.0 <= value <= 1.0, f"{attr} should be between 0 and 1"
    
    print("✅ Spirits defaults test passed")


def test_core_initialization():
    """Test ElysiaAvatarCore initialization"""
    core = ElysiaAvatarCore()
    assert core.expression is not None, "Expression should be initialized"
    assert core.spirits is not None, "Spirits should be initialized"
    assert core.beat_phase == 0.0, "Beat phase should start at 0"
    print("✅ Core initialization test passed")


def test_beat_update():
    """Test heartbeat animation update"""
    core = ElysiaAvatarCore()
    initial_phase = core.beat_phase
    
    core.update_beat(0.1)  # 100ms
    
    assert core.beat_phase != initial_phase, "Beat phase should change"
    assert 0.0 <= core.expression.beat <= 1.0, "Beat should be in [0, 1]"
    print("✅ Beat update test passed")


def test_state_message():
    """Test state message generation"""
    core = ElysiaAvatarCore()
    state = core.get_state_message()
    
    assert "expression" in state, "State should contain expression"
    assert "spirits" in state, "State should contain spirits"
    
    # Check expression fields
    expr = state["expression"]
    required_expr_fields = ["mouth_curve", "eye_open", "brow_furrow", "beat", "mouth_width"]
    for field in required_expr_fields:
        assert field in expr, f"Expression should have {field}"
    
    # Check spirits fields
    spirits = state["spirits"]
    required_spirit_fields = ["fire", "water", "earth", "air", "light", "dark", "aether"]
    for field in required_spirit_fields:
        assert field in spirits, f"Spirits should have {field}"
    
    print("✅ State message test passed")


def test_expression_ranges():
    """Test that expression values stay in valid ranges"""
    core = ElysiaAvatarCore()
    
    # Update multiple times
    for i in range(10):
        core.update_beat(0.033)  # ~30 FPS
        core.update_expression_from_emotion()
    
    # Check ranges
    assert -1.0 <= core.expression.mouth_curve <= 1.0, "mouth_curve out of range"
    assert 0.0 <= core.expression.eye_open <= 1.0, "eye_open out of range"
    assert 0.0 <= core.expression.brow_furrow <= 1.0, "brow_furrow out of range"
    assert 0.0 <= core.expression.beat <= 1.0, "beat out of range"
    
    print("✅ Expression ranges test passed")


def test_spirit_ranges():
    """Test that spirit values stay in valid ranges"""
    core = ElysiaAvatarCore()
    
    # Update multiple times
    for i in range(10):
        core.update_spirits_from_emotion()
    
    # Check all spirits are in range
    for attr in ['fire', 'water', 'earth', 'air', 'light', 'dark', 'aether']:
        value = getattr(core.spirits, attr)
        assert 0.0 <= value <= 1.0, f"{attr} out of range: {value}"
    
    print("✅ Spirit ranges test passed")


def test_full_update_cycle():
    """Test a complete update cycle"""
    core = ElysiaAvatarCore()
    
    # Simulate several frames
    for frame in range(30):  # 1 second at 30 FPS
        delta_time = 1.0 / 30.0
        
        core.update_beat(delta_time)
        core.update_expression_from_emotion()
        core.update_spirits_from_emotion()
        
        state = core.get_state_message()
        assert state is not None, f"State should be valid at frame {frame}"
    
    print("✅ Full update cycle test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("  Avatar Server Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_expression_defaults,
        test_spirits_defaults,
        test_core_initialization,
        test_beat_update,
        test_state_message,
        test_expression_ranges,
        test_spirit_ranges,
        test_full_update_cycle,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} error: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
