"""
Unit Tests for Avatar Server Core Functionality
================================================

Tests for:
- Delta update calculation
- Adaptive FPS calculation
- Expression and Spirit state management
- Message queue and serialization
"""

import sys
from pathlib import Path
import asyncio
import time

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.Interface.avatar_server import (
    Expression,
    Spirits,
    ElysiaAvatarCore,
    AvatarWebSocketServer
)


class TestExpression:
    """Test Expression dataclass"""
    
    def test_default_values(self):
        """Test default expression values"""
        expr = Expression()
        assert expr.mouth_curve == 0.0
        assert expr.eye_open == 1.0
        assert expr.brow_furrow == 0.0
        assert expr.beat == 0.0
        assert expr.mouth_width == 0.0
        print("✓ Expression default values correct")
    
    def test_custom_values(self):
        """Test custom expression values"""
        expr = Expression(mouth_curve=0.5, eye_open=0.8, brow_furrow=0.2)
        assert expr.mouth_curve == 0.5
        assert expr.eye_open == 0.8
        assert expr.brow_furrow == 0.2
        print("✓ Expression custom values correct")
    
    def test_value_ranges(self):
        """Test expression value ranges"""
        expr = Expression(mouth_curve=1.5, eye_open=-0.5)
        # Values should be set (no clamping in dataclass)
        assert expr.mouth_curve == 1.5
        assert expr.eye_open == -0.5
        print("✓ Expression accepts any float values")


class TestSpirits:
    """Test Spirits dataclass"""
    
    def test_default_values(self):
        """Test default spirit values"""
        spirits = Spirits()
        assert spirits.fire == 0.1
        assert spirits.water == 0.1
        assert spirits.earth == 0.3
        assert spirits.air == 0.2
        assert spirits.light == 0.2
        assert spirits.dark == 0.1
        assert spirits.aether == 0.1
        print("✓ Spirits default values correct")
    
    def test_all_in_valid_range(self):
        """Test that default spirits are in 0-1 range"""
        spirits = Spirits()
        for attr in ['fire', 'water', 'earth', 'air', 'light', 'dark', 'aether']:
            value = getattr(spirits, attr)
            assert 0.0 <= value <= 1.0, f"{attr} out of range: {value}"
        print("✓ All spirit values in valid range (0-1)")


class TestElysiaAvatarCore:
    """Test ElysiaAvatarCore functionality"""
    
    def test_initialization(self):
        """Test core initialization"""
        core = ElysiaAvatarCore()
        assert core.expression is not None
        assert core.spirits is not None
        assert core.beat_phase == 0.0
        assert core.last_state is None
        assert core.delta_threshold == 0.01
        print("✓ Core initialization successful")
    
    def test_get_state_message(self):
        """Test full state message generation"""
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
        
        print("✓ State message contains all required fields")
    
    def test_delta_update_full_first(self):
        """Test that first delta message is full state"""
        core = ElysiaAvatarCore()
        
        msg = core.get_delta_message()
        assert msg is not None
        assert msg['type'] == 'full'
        assert 'expression' in msg
        assert 'spirits' in msg
        print("✓ First delta message is full state")
    
    def test_delta_update_no_change(self):
        """Test that no changes return None"""
        core = ElysiaAvatarCore()
        
        # First call (full)
        core.get_delta_message()
        
        # Second call with no changes
        msg = core.get_delta_message()
        assert msg is None
        print("✓ No changes return None (skips transmission)")
    
    def test_delta_update_with_change(self):
        """Test that changes trigger delta update"""
        core = ElysiaAvatarCore()
        
        # First call (full)
        core.get_delta_message()
        
        # Make a significant change
        core.expression.mouth_curve = 0.5
        
        msg = core.get_delta_message()
        assert msg is not None
        assert msg['type'] == 'delta'
        assert 'expression' in msg
        assert 'mouth_curve' in msg['expression']
        assert msg['expression']['mouth_curve'] == 0.5
        print("✓ Changes trigger delta update with only changed fields")
    
    def test_delta_update_threshold(self):
        """Test that small changes below threshold are ignored"""
        core = ElysiaAvatarCore()
        
        # First call (full)
        core.get_delta_message()
        
        # Make a change below threshold (0.01)
        core.expression.mouth_curve = 0.005
        
        msg = core.get_delta_message()
        assert msg is None
        print("✓ Changes below threshold (0.01) are ignored")
    
    def test_delta_update_spirits_change(self):
        """Test delta update for spirits changes"""
        core = ElysiaAvatarCore()
        
        # First call (full)
        core.get_delta_message()
        
        # Change only spirits
        core.spirits.fire = 0.8
        
        msg = core.get_delta_message()
        assert msg is not None
        assert msg['type'] == 'delta'
        assert 'spirits' in msg
        assert 'fire' in msg['spirits']
        assert msg['spirits']['fire'] == 0.8
        assert 'expression' not in msg  # Expression didn't change
        print("✓ Spirits changes tracked independently")
    
    def test_update_beat(self):
        """Test beat animation update"""
        core = ElysiaAvatarCore()
        initial_phase = core.beat_phase
        
        # Update beat
        core.update_beat(0.1)  # 100ms
        
        assert core.beat_phase != initial_phase
        assert 0.0 <= core.expression.beat <= 1.0
        print("✓ Beat animation updates correctly")


class TestAvatarWebSocketServer:
    """Test AvatarWebSocketServer functionality"""
    
    def test_initialization(self):
        """Test server initialization"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = AvatarWebSocketServer(port=8765)
        assert server.core is not None
        assert server.min_fps == 15
        assert server.max_fps == 60
        assert server.activity_level == 0.0
        print("✓ Server initialization successful")
    
    def test_adaptive_fps_idle(self):
        """Test adaptive FPS in idle state"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = AvatarWebSocketServer(port=8765)
        fps = server.calculate_adaptive_fps()
        
        assert 15 <= fps <= 20, f"Idle FPS should be 15-20, got {fps}"
        print(f"✓ Idle FPS: {fps} (within expected range)")
    
    def test_adaptive_fps_recent_message(self):
        """Test adaptive FPS with recent message"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = AvatarWebSocketServer(port=8765)
        
        # Simulate recent message
        server.last_message_time = time.time()
        fps = server.calculate_adaptive_fps()
        
        assert fps > 25, f"Active FPS should be > 25, got {fps}"
        print(f"✓ Active FPS (recent message): {fps}")
    
    def test_adaptive_fps_multiple_clients(self):
        """Test adaptive FPS with multiple clients"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = AvatarWebSocketServer(port=8765)
        
        # Add fake clients
        server.clients.add("client1")
        server.clients.add("client2")
        server.clients.add("client3")
        
        fps_idle = server.calculate_adaptive_fps()
        
        # Add recent message
        server.last_message_time = time.time()
        fps_active = server.calculate_adaptive_fps()
        
        assert fps_active > fps_idle
        assert fps_active >= 35, f"Active FPS with clients should be >= 35, got {fps_active}"
        print(f"✓ Active FPS (3 clients + message): {fps_active}")
    
    def test_adaptive_fps_ranges(self):
        """Test that adaptive FPS stays within bounds"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = AvatarWebSocketServer(port=8765)
        
        # Test various scenarios
        scenarios = []
        
        # Idle
        fps_idle = server.calculate_adaptive_fps()
        scenarios.append(("idle", fps_idle))
        
        # Recent message
        server.last_message_time = time.time()
        fps_message = server.calculate_adaptive_fps()
        scenarios.append(("recent_message", fps_message))
        
        # Many clients
        for i in range(15):
            server.clients.add(f"client{i}")
        fps_many = server.calculate_adaptive_fps()
        scenarios.append(("many_clients", fps_many))
        
        # Verify all in range
        for scenario, fps in scenarios:
            assert server.min_fps <= fps <= server.max_fps, \
                f"{scenario}: FPS {fps} out of range [{server.min_fps}, {server.max_fps}]"
        
        print(f"✓ All FPS values within bounds: {scenarios}")


def run_all_tests():
    """Run all test suites"""
    print("=" * 60)
    print("Running Avatar Server Unit Tests")
    print("=" * 60)
    
    test_classes = [
        TestExpression,
        TestSpirits,
        TestElysiaAvatarCore,
        TestAvatarWebSocketServer
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 60)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, test_method)
                method()
                passed_tests += 1
            except Exception as e:
                failed_tests.append((test_class.__name__, test_method, e))
                print(f"✗ {test_method}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print("=" * 60)
    
    if failed_tests:
        print("\n❌ Failed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print("\n✅ All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
