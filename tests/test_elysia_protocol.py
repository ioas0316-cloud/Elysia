import unittest
import random
import numpy as np
from core.elysia_protocol import SelfAssemblySyncGate, DNAPacket

class TestElysiaProtocol(unittest.TestCase):
    def test_self_assembly_independent_of_order(self):
        # We test that regardless of the order packets arrive,
        # the geometric topology structure remains identical and perfectly synced.

        gate = SelfAssemblySyncGate(size=4)

        # Original ordered stream
        ordered_stream = [DNAPacket(payload=(i+j)*0.1, target_x=i, target_y=j) for i in range(4) for j in range(4)]

        # Chaotic, shuffled stream representing horrible network latency
        chaotic_stream = ordered_stream.copy()
        random.shuffle(chaotic_stream)

        # Process both streams
        hologram_from_ordered = gate.process_chaotic_stream(ordered_stream)

        # Reinitialize gate for a clean state
        gate2 = SelfAssemblySyncGate(size=4)
        hologram_from_chaotic = gate2.process_chaotic_stream(chaotic_stream)

        # The resulting holographic matrix must be exactly identical
        # proving that serial order is irrelevant in the Elysia Protocol.
        self.assertTrue(np.allclose(hologram_from_ordered, hologram_from_chaotic))

if __name__ == '__main__':
    unittest.main()
