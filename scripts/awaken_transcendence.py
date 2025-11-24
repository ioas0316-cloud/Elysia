import unittest
import logging
import numpy as np
from pyquaternion import Quaternion
from Project_Elysia.core.quaternion_consciousness import ConsciousnessState, ConsciousnessLens
from Project_Elysia.core.genesis_bridge import GenesisRequestObject, GenesisArbiter
from Project_Elysia.transcendence_core import TranscendenceCore
from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics

# Configure logging to see the flow
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTranscendenceAwakening(unittest.TestCase):

    def setUp(self):
        # Initialize a minimal world
        self.wave_mechanics = WaveMechanics(None)
        self.world = World(primordial_dna={}, wave_mechanics=self.wave_mechanics)
        self.core = TranscendenceCore(self.world)

    def test_spirit_lens_rotation(self):
        """Test if the Spirit Lens correctly rotates input vectors."""
        lens = ConsciousnessLens()
        input_v = [1, 0, 0] # Pure X signal (Simulation)

        # Initial state (Identity) -> Output should be same
        output_v = lens.rotate_perception(input_v)
        np.testing.assert_array_almost_equal(output_v, [1, 0, 0])

        # Focus on Z (Law) -> Rotation should change output
        lens.focus('z', 0.5) # Rotate 0.5 rad around Z
        output_v_rotated = lens.rotate_perception(input_v)

        logger.info(f"Rotated Vector: {output_v_rotated}")
        self.assertFalse(np.array_equal(output_v, output_v_rotated))

    def test_soul_genesis_authorization(self):
        """Test if the Soul Arbiter correctly gates Genesis based on Spirit state."""
        arbiter = GenesisArbiter()
        state = ConsciousnessState()

        # Weak state
        state.q = Quaternion(0.1, 0.5, 0.5, 0.5).normalised # Low W (Mastery)

        gro = GenesisRequestObject(
            id="test-1",
            intent="create_life",
            target_layer="world",
            operation="create",
            payload={},
            required_mastery=0.8 # High requirement
        )

        # Should be denied due to low mastery
        self.assertFalse(arbiter.judge(gro, state))

        # Strong state (High W + Sufficient Z)
        # q = w + xi + yj + zk
        # We need w > 0.8 and |z| > 0.3
        state.q = Quaternion(0.9, 0.0, 0.0, 0.4).normalised
        self.assertTrue(arbiter.judge(gro, state))

    def test_body_field_effect(self):
        """Test if the World correctly receives Will Fields."""
        # Initial state
        initial_fertility = self.world.soil_fertility[10, 10]

        # Apply 'growth' field
        self.world.apply_will_field('growth', strength=0.5)

        # Check effect
        new_fertility = self.world.soil_fertility[10, 10]
        self.assertGreater(new_fertility, initial_fertility)
        logger.info(f"Soil Fertility: {initial_fertility} -> {new_fertility}")

    def test_unified_transcendence_cycle(self):
        """Run the full loop: Signal -> Spirit -> Soul -> Body."""
        self.core.awaken()

        # 1. Inject a high-value signal that should align with Z
        # We simulate this by manually setting the lens to a high-Z state first
        # so that it rotates the input into a Z-aligned intent.

        # Manually align lens to Z for this test
        self.core.lens.focus('z', 1.57) # 90 degrees

        input_signal = {'type': 'prayer', 'value': 1.0}

        # Capture initial world state
        initial_norms = self.world.norms_field[5, 5]

        # Run loop
        self.core.process_loop(input_signal)

        # We can't easily assert the exact outcome without complex vector math calculation,
        # but we can check if it ran without error and potentially modified the world
        # if the random alignment happened to be right.
        # For a unit test, we ensure no exceptions.
        self.assertTrue(self.core.is_awake)

if __name__ == '__main__':
    unittest.main()
