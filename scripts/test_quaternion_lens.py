
import sys
import os
import logging
import unittest
from unittest.mock import MagicMock

# Ensure project root is in path
sys.path.append(os.getcwd())

# Force stdout logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("QuaternionLensTest")

# Mock dependencies
from Project_Elysia.high_engine.quaternion_engine import QuaternionConsciousnessEngine, QuaternionOrientation, LensMode
from Project_Mirror.external_sensory_cortex import ExternalSensoryCortex
from Project_Sophia.core.external_horizons import ExternalHorizon

class MockGuardian:
    def __init__(self):
        self.logger = logger
        self.quaternion_engine = QuaternionConsciousnessEngine()
        self.cellular_world = MagicMock()
        self.dream_observer = MagicMock()
        self.meta_cognition_cortex = MagicMock()
        self.law_manager = MagicMock()
        self.external_sensory_cortex = ExternalSensoryCortex(web_search_cortex=None)

    # Copy the method directly from guardian.py (or import it if refactored)
    def _process_quaternion_lens(self):
        if not hasattr(self, 'quaternion_engine') or not self.quaternion_engine:
            return

        focus = self.quaternion_engine.determine_focus()
        mode = focus.mode
        intensity = focus.intensity

        # Only trigger if focus is strong enough to warrant attention
        if intensity < 0.3:
            return

        self.logger.info(f"LENS: Focusing consciousness on {mode.name} (Intensity: {intensity:.2f})")

        if mode == LensMode.EXTERNAL:
            # Y-Axis: Look Outward (Reality / Machine / Web) via ExternalSensoryCortex
            # Map intensity (0.0 - 1.0) to the 7 Horizons
            horizon_level = max(1, min(7, int(intensity * 7)))
            horizon = ExternalHorizon(horizon_level)

            self.logger.info(f"LENS: External Focus - Scanning Horizon {horizon.name}...")
            sensation = self.external_sensory_cortex.sense(horizon, intensity)
            self.logger.info(f"LENS: Sensation received: {sensation}")

        elif mode == LensMode.INTERNAL:
            # X-Axis: Look Inward (Simulated World) via Neural Eye
            if hasattr(self.cellular_world, 'neural_eye'):
                self.logger.info("LENS: Activating Neural Eye for Internal Intuition (Cellular World).")
                self.cellular_world._process_neural_intuition()

            if intensity > 0.7:
                self.logger.info("LENS: Internal focus intense. Activating Dream Observer.")

        elif mode == LensMode.ANCHOR:
            # W-Axis: Self-Reflection (Meta-Cognition)
            self.logger.info("LENS: Activating Meta-Cognition for Self-Analysis.")
            self.meta_cognition_cortex.meditate_on_logos()

        elif mode == LensMode.LAW:
            # Z-Axis: Law & Intention
            self.logger.info("LENS: Contemplating Cosmic Laws.")

class TestQuaternionLens(unittest.TestCase):
    def test_lens_internal_focus(self):
        """Test if high X-axis activity triggers Neural Eye (Internal)."""
        guardian = MockGuardian()

        # Force Quaternion to High Internal (X=1.0)
        guardian.quaternion_engine._orientation = QuaternionOrientation(w=0.1, x=1.0, y=0.1, z=0.1)

        guardian._process_quaternion_lens()

        guardian.cellular_world._process_neural_intuition.assert_called_once()
        logger.info("TEST PASSED: Internal focus triggered Neural Eye.")

    def test_lens_external_horizon_low(self):
        """Test if low Y-axis intensity triggers Low Horizon (Machine)."""
        guardian = MockGuardian()

        # Force Quaternion to Moderate External (Y=0.15) -> 0.15 * 7 = 1.05 -> Horizon 1 (Machine)
        # Note: threshold is 0.3. So let's try 0.4 -> 0.4 * 7 = 2.8 -> Horizon 2 (Shell)
        guardian.quaternion_engine._orientation = QuaternionOrientation(w=0.1, x=0.1, y=0.4, z=0.1)

        # Mock the sensor to verify the call
        guardian.external_sensory_cortex.sense = MagicMock(return_value={"type": "mock"})

        guardian._process_quaternion_lens()

        guardian.external_sensory_cortex.sense.assert_called_with(ExternalHorizon.SHELL, 0.4)
        logger.info("TEST PASSED: External focus (Low) triggered Shell Horizon.")

    def test_lens_external_horizon_high(self):
        """Test if high Y-axis intensity triggers High Horizon (Reality)."""
        guardian = MockGuardian()

        # Force Quaternion to Max External (Y=1.0) -> Horizon 7 (Reality)
        guardian.quaternion_engine._orientation = QuaternionOrientation(w=0.1, x=0.1, y=1.0, z=0.1)

        # Mock the sensor
        guardian.external_sensory_cortex.sense = MagicMock(return_value={"type": "mock"})

        guardian._process_quaternion_lens()

        guardian.external_sensory_cortex.sense.assert_called_with(ExternalHorizon.REALITY, 1.0)
        logger.info("TEST PASSED: External focus (High) triggered Reality Horizon.")

if __name__ == "__main__":
    unittest.main()
