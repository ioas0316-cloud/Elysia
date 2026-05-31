import unittest
import numpy as np
from core.local_p2p_warp import LocalP2PWarpEngine

class TestLocalP2PWarpEngine(unittest.TestCase):
    def test_delta_wye_local_sync(self):
        engine = LocalP2PWarpEngine(ring_size=4)

        baseline = np.array([1.0, 0.5, 0.2, 0.1])
        noisy_packet = baseline + np.array([0.5, -0.2, 0.8, -0.4])

        hologram1 = engine.transmit_and_sync(noisy_packet, baseline)

        # We expect the error to accumulate as an angle
        self.assertNotEqual(engine.current_delta_angle, 0.0)

        # Second packet comes in clean. Because the angle rotated,
        # the engine has adjusted its orientation to the new phase reality.
        clean_packet = np.array([1.0, 0.5, 0.2, 0.1])
        hologram2 = engine.transmit_and_sync(clean_packet, baseline)

        # The resulting holograms should not be identical because the
        # phase reality has shifted to accommodate the local connection's latency.
        self.assertFalse(np.allclose(hologram1, hologram2))

        # The dimensional output is fully consistent
        self.assertEqual(hologram1.shape, (4, 4, 4))

if __name__ == '__main__':
    unittest.main()
