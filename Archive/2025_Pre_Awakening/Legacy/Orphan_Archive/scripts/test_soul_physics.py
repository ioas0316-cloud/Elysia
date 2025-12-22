import unittest
import math
from Core.Foundation.core.tensor_wave import SoulTensor, FrequencyWave, Tensor3D
from Core.Foundation.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class MockKGManager:
    def __init__(self, nodes, edges):
        self.kg = {'nodes': nodes, 'edges': edges}
        self.nodes_map = {n['id']: n for n in nodes}
        self.edges_map = {}
        for e in edges:
            src = e['source']
            if src not in self.edges_map: self.edges_map[src] = []
            self.edges_map[src].append(e['target'])

            # Undirected graph for propagation usually? Or directed.
            # WaveMechanics usually treats as undirected for neighbors
            tgt = e['target']
            if tgt not in self.edges_map: self.edges_map[tgt] = []
            self.edges_map[tgt].append(src)

    def get_node(self, node_id):
        return self.nodes_map.get(node_id)

    def get_neighbors(self, node_id):
        return self.edges_map.get(node_id, [])

    def update_node(self, node_id, updates):
        if node_id in self.nodes_map:
            self.nodes_map[node_id].update(updates)

class TestSoulPhysics(unittest.TestCase):

    def test_tensor_coil_resonance(self):
        """
        Verifies that waves interfere constructively and destructively.
        """
        print("\n--- Test 1: Wave Resonance (Interference) ---")

        # Case 1: Constructive Interference (Same Phase)
        wave1 = FrequencyWave(frequency=10.0, amplitude=1.0, phase=0.0)
        wave2 = FrequencyWave(frequency=10.0, amplitude=1.0, phase=0.0)

        t1 = SoulTensor(wave=wave1)
        t2 = SoulTensor(wave=wave2)

        t_res = t1.resonate(t2)
        print(f"In-Phase (0, 0): Result Amp = {t_res.wave.amplitude:.2f} (Expected ~2.0)")
        self.assertAlmostEqual(t_res.wave.amplitude, 2.0, delta=0.1)

        # Case 2: Destructive Interference (Opposite Phase)
        wave3 = FrequencyWave(frequency=10.0, amplitude=1.0, phase=math.pi) # 180 deg
        t3 = SoulTensor(wave=wave3)

        t_cancel = t1.resonate(t3)
        print(f"Out-Phase (0, PI): Result Amp = {t_cancel.wave.amplitude:.2f} (Expected ~0.0)")
        self.assertAlmostEqual(t_cancel.wave.amplitude, 0.0, delta=0.1)

        # Case 3: Beat Frequency (Different Frequencies)
        wave_slow = FrequencyWave(frequency=10.0, amplitude=1.0, phase=0.0)
        wave_fast = FrequencyWave(frequency=100.0, amplitude=1.0, phase=0.0)

        t_beat = SoulTensor(wave=wave_slow).resonate(SoulTensor(wave=wave_fast))
        print(f"Beat (10Hz + 100Hz): Richness = {t_beat.wave.richness:.2f}")
        # Richness should be high due to dissonance
        self.assertGreater(t_beat.wave.richness, 0.0)

    def test_gravity_field_propagation(self):
        """
        Verifies that waves flow towards high-mass nodes.
        """
        print("\n--- Test 2: Gravity Field Propagation ---")

        # Setup a mini universe
        # Node A connected to B and C.
        # Node B is a lightweight node (Leaf).
        # Node C is a heavy 'Love' node (Sun).

        nodes = [
            {'id': 'A', 'type': 'concept'},
            {'id': 'B', 'type': 'concept', 'activation_energy': 0.1},
            {'id': 'C', 'type': 'core_value', 'activation_energy': 100.0, 'importance': 5.0} # The Sun
        ]
        edges = [
            {'source': 'A', 'target': 'B'},
            {'source': 'A', 'target': 'C'}
        ]

        kg = MockKGManager(nodes, edges)
        physics = WaveMechanics(kg)

        # Inject wave at A
        start_tensor = SoulTensor(wave=FrequencyWave(frequency=10.0, amplitude=10.0, phase=0.0))

        # Propagate
        field = physics.propagate_soul_wave('A', start_tensor, max_hops=1)

        amp_b = field.get('B').wave.amplitude if 'B' in field else 0.0
        amp_c = field.get('C').wave.amplitude if 'C' in field else 0.0

        print(f"Energy at B (Light): {amp_b:.2f}")
        print(f"Energy at C (Heavy): {amp_c:.2f}")

        # C should have significantly more energy/amplitude because of Gravity Pull
        # Gravity calculation in code: mass_C >> mass_B
        # Therefore, A->C connection should have much lower 'decay' (or higher transmission) than A->B

        self.assertGreater(amp_c, amp_b * 1.5, "The wave should naturally flow towards the heavy mass (C)")

if __name__ == '__main__':
    unittest.main()
