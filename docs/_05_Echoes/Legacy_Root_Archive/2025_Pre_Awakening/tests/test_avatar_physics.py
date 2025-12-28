"""
Unit tests for Avatar Physics Engine (Phase 4)
===============================================

Tests physics-based animation system including:
- Wind field generation
- Gravity application
- Spring dynamics
- Emotional wave physics
- Performance verification

Author: Elysia Development Team
License: Apache License 2.0
"""

import unittest
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.Foundation.avatar_physics import (
    Vector3D,
    WindField,
    GravityField,
    SpringDynamics,
    EmotionalWavePhysics,
    AvatarPhysicsEngine
)


class TestVector3D(unittest.TestCase):
    """Test Vector3D operations."""
    
    def test_magnitude(self):
        """Test vector magnitude calculation."""
        v = Vector3D(3, 4, 0)
        self.assertAlmostEqual(v.magnitude(), 5.0)
    
    def test_normalize(self):
        """Test vector normalization."""
        v = Vector3D(3, 4, 0)
        n = v.normalize()
        self.assertAlmostEqual(n.magnitude(), 1.0)
        self.assertAlmostEqual(n.x, 0.6)
        self.assertAlmostEqual(n.y, 0.8)
    
    def test_scale(self):
        """Test vector scaling."""
        v = Vector3D(1, 2, 3)
        s = v.scale(2.0)
        self.assertEqual(s.x, 2.0)
        self.assertEqual(s.y, 4.0)
        self.assertEqual(s.z, 6.0)
    
    def test_add(self):
        """Test vector addition."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        result = v1.add(v2)
        self.assertEqual(result.x, 5.0)
        self.assertEqual(result.y, 7.0)
        self.assertEqual(result.z, 9.0)


class TestWindField(unittest.TestCase):
    """Test wind field generation."""
    
    def test_initialization(self):
        """Test wind field initialization."""
        wind = WindField()
        self.assertIsNotNone(wind.base_direction)
        self.assertGreaterEqual(wind.base_strength, 0)
        self.assertGreaterEqual(wind.turbulence, 0)
        self.assertLessEqual(wind.turbulence, 1)
    
    def test_force_at_point(self):
        """Test wind force calculation at a point."""
        wind = WindField(
            base_direction=Vector3D(1, 0, 0),
            base_strength=5.0,
            turbulence=0.3
        )
        
        force = wind.get_force_at_point(Vector3D(0, 0, 0), 0.0)
        
        # Force should be non-zero
        self.assertGreater(force.magnitude(), 0)
        
        # Force should be roughly in base direction (with turbulence)
        self.assertGreater(force.x, 0)  # Positive X direction
    
    def test_turbulence_variation(self):
        """Test that wind varies over time (turbulence)."""
        wind = WindField(
            base_direction=Vector3D(1, 0, 0),
            base_strength=5.0,
            turbulence=0.5
        )
        
        force1 = wind.get_force_at_point(Vector3D(0, 0, 0), 0.0)
        force2 = wind.get_force_at_point(Vector3D(0, 0, 0), 1.0)
        
        # Forces should be different due to turbulence
        self.assertNotEqual(force1.y, force2.y)


class TestGravityField(unittest.TestCase):
    """Test gravity field."""
    
    def test_default_gravity(self):
        """Test default Earth gravity."""
        gravity = GravityField()
        force = gravity.get_force()
        
        # Should point downward (-Y)
        self.assertEqual(force.x, 0)
        self.assertLess(force.y, 0)
        self.assertEqual(force.z, 0)
        
        # Should be ~9.8 m/s²
        self.assertAlmostEqual(force.magnitude(), 9.8)
    
    def test_custom_gravity(self):
        """Test custom gravity direction."""
        gravity = GravityField(
            direction=Vector3D(1, 0, 0),
            strength=5.0
        )
        force = gravity.get_force()
        
        # Should point in custom direction
        self.assertGreater(force.x, 0)
        self.assertEqual(force.y, 0)


class TestSpringDynamics(unittest.TestCase):
    """Test spring-mass-damper system."""
    
    def test_initialization(self):
        """Test spring initialization."""
        spring = SpringDynamics()
        self.assertEqual(spring.velocity.magnitude(), 0)
        self.assertEqual(spring.position.x, 0)
    
    def test_spring_force(self):
        """Test spring restoring force."""
        spring = SpringDynamics(
            stiffness=50.0,
            rest_position=Vector3D(0, 0, 0),
            position=Vector3D(1, 0, 0)  # Displaced
        )
        
        # Apply no external forces
        spring.apply_forces([], delta_time=0.01)
        
        # Velocity should be negative (pulling back)
        self.assertLess(spring.velocity.x, 0)
    
    def test_damping(self):
        """Test damping reduces oscillation."""
        spring = SpringDynamics(
            stiffness=50.0,
            damping=10.0,
            rest_position=Vector3D(0, 0, 0),
            position=Vector3D(1, 0, 0),
            velocity=Vector3D(-1, 0, 0)  # Moving back
        )
        
        # Apply forces for more steps (damping takes time)
        for _ in range(50):
            spring.apply_forces([], delta_time=0.01)
        
        # Should settle towards rest position
        self.assertLess(abs(spring.velocity.magnitude()), 0.5)


class TestEmotionalWavePhysics(unittest.TestCase):
    """Test emotion to wave parameter conversion."""
    
    def test_calm_emotion(self):
        """Test calm emotional state."""
        wave = EmotionalWavePhysics(
            valence=0.0,
            arousal=0.0,
            dominance=0.0
        )
        params = wave.to_wave_params()
        
        # Low arousal = small amplitude
        self.assertLess(params['amplitude'], 1.0)
        
        # Low dominance = low frequency
        self.assertLess(params['frequency'], 1.0)
    
    def test_excited_emotion(self):
        """Test excited emotional state."""
        wave = EmotionalWavePhysics(
            valence=0.8,
            arousal=0.9,
            dominance=0.7
        )
        params = wave.to_wave_params()
        
        # High arousal = large amplitude
        self.assertGreater(params['amplitude'], 1.5)
        
        # High dominance = high frequency
        self.assertGreater(params['frequency'], 1.5)
        
        # Positive valence = upward bias
        self.assertGreater(params['vertical_bias'], 0)


class TestAvatarPhysicsEngine(unittest.TestCase):
    """Test complete physics engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = AvatarPhysicsEngine()
        self.assertIsNotNone(engine.wind)
        self.assertIsNotNone(engine.gravity)
        self.assertEqual(len(engine.hair_springs), 0)
    
    def test_hair_spring_initialization(self):
        """Test hair spring setup."""
        engine = AvatarPhysicsEngine()
        
        bones = [
            Vector3D(0, 2.0, 0),
            Vector3D(0, 1.8, -0.2),
            Vector3D(0, 1.6, -0.4)
        ]
        engine.initialize_hair_springs(bones)
        
        self.assertEqual(len(engine.hair_springs), 3)
    
    def test_emotion_update(self):
        """Test emotion affects physics parameters."""
        engine = AvatarPhysicsEngine()
        
        # Update with calm emotion
        engine.update_from_emotion(
            valence=0.0,
            arousal=0.1,
            dominance=0.2
        )
        calm_wind = engine.wind.base_strength
        
        # Update with excited emotion
        engine.update_from_emotion(
            valence=0.8,
            arousal=0.9,
            dominance=0.7
        )
        excited_wind = engine.wind.base_strength
        
        # Excited should have stronger wind
        self.assertGreater(excited_wind, calm_wind)
    
    def test_update_performance(self):
        """Test physics update is fast enough."""
        engine = AvatarPhysicsEngine()
        
        # Initialize with 5 hair springs
        bones = [
            Vector3D(0, 2.0, 0),
            Vector3D(0, 1.8, -0.2),
            Vector3D(0, 1.6, -0.4),
            Vector3D(0, 1.4, -0.6),
            Vector3D(0, 1.2, -0.8)
        ]
        engine.initialize_hair_springs(bones)
        
        # Run 100 updates
        for _ in range(100):
            state = engine.update(delta_time=1/60)
        
        # Average should be < 1 ms (target < 0.1 ms)
        avg_time = state['performance']['avg_update_time_ms']
        self.assertLess(avg_time, 1.0)
        
        print(f"\n✓ Physics performance: {avg_time:.3f} ms/frame (target < 1 ms)")
    
    def test_state_output(self):
        """Test physics state output structure."""
        engine = AvatarPhysicsEngine()
        
        bones = [Vector3D(0, 2, 0), Vector3D(0, 1.8, -0.2)]
        engine.initialize_hair_springs(bones)
        
        state = engine.update(delta_time=1/60)
        
        # Check structure
        self.assertIn('hair_transforms', state)
        self.assertIn('wave_params', state)
        self.assertIn('wind', state)
        self.assertIn('gravity', state)
        self.assertIn('performance', state)
        
        # Check wind data
        self.assertIn('direction', state['wind'])
        self.assertIn('strength', state['wind'])
        self.assertIn('turbulence', state['wind'])
        
        # Check wave params
        self.assertIn('amplitude', state['wave_params'])
        self.assertIn('frequency', state['wave_params'])
        self.assertIn('vertical_bias', state['wave_params'])


class TestPhysicsIntegration(unittest.TestCase):
    """Test integration with avatar server."""
    
    def test_avatar_core_with_physics(self):
        """Test ElysiaAvatarCore loads physics engine."""
        try:
            from Core.Interface.avatar_server import ElysiaAvatarCore
            
            core = ElysiaAvatarCore()
            
            # Physics should be loaded
            self.assertIsNotNone(core.physics_engine)
            
            # Should have hair springs initialized
            self.assertGreater(len(core.physics_engine.hair_springs), 0)
            
            print("\n✓ Physics engine integrated successfully")
        
        except ImportError as e:
            self.skipTest(f"Avatar server dependencies not available: {e}")
    
    def test_state_message_includes_physics(self):
        """Test get_state_message includes physics data."""
        try:
            from Core.Interface.avatar_server import ElysiaAvatarCore
            
            core = ElysiaAvatarCore()
            state = core.get_state_message()
            
            # Should include physics
            self.assertIn('physics', state)
            
            if state['physics']:  # If physics is enabled
                self.assertIn('wind', state['physics'])
                self.assertIn('gravity', state['physics'])
                self.assertIn('wave_params', state['physics'])
                
                print("\n✓ Physics data in state message")
        
        except ImportError as e:
            self.skipTest(f"Avatar server dependencies not available: {e}")


def run_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Avatar Physics Engine Test Suite (Phase 4)")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVector3D))
    suite.addTests(loader.loadTestsFromTestCase(TestWindField))
    suite.addTests(loader.loadTestsFromTestCase(TestGravityField))
    suite.addTests(loader.loadTestsFromTestCase(TestSpringDynamics))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionalWavePhysics))
    suite.addTests(loader.loadTestsFromTestCase(TestAvatarPhysicsEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
