
import unittest
import shutil
import os
import sys
from unittest.mock import MagicMock

# --- MOCKING DEPENDENCIES ---
# We must mock google.generativeai BEFORE importing modules that depend on it.
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

# Also mock gemini_api to safely import vector_utils
gemini_mock = MagicMock()
gemini_mock.get_text_embedding.return_value = [0.1] * 768
sys.modules['Project_Sophia.gemini_api'] = gemini_mock

from Core.Foundation.wave_mechanics import WaveMechanics
from Core.Foundation.core.tensor_wave import FrequencyWave, SoulTensor, Tensor3D
from Core.Foundation.core.world import World
from tools.kg_manager import KGManager

class TestQuantumMechanics(unittest.TestCase):
    def setUp(self):
        # Setup temporary KG
        self.test_dir = "test_quantum_data"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        self.kg_path = os.path.join(self.test_dir, "kg.json")
        self.kg_manager = KGManager(self.kg_path)
        # Create dummy nodes
        self.kg_manager.add_node("NodeA", {"label": "A"})
        self.kg_manager.add_node("NodeB", {"label": "B"})
        self.kg_manager.add_node("NodeC", {"label": "C", "activation_energy": 0.0})
        self.kg_manager.save()

        self.wave_mechanics = WaveMechanics(self.kg_manager)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_entanglement(self):
        print("\n--- Testing Quantum Entanglement ---")
        # 1. Entangle A and B
        self.wave_mechanics.entangle_nodes("NodeA", "NodeB")

        # 2. Verify shared state existence
        tensor_a = self.wave_mechanics.get_node_tensor("NodeA")
        tensor_b = self.wave_mechanics.get_node_tensor("NodeB")

        self.assertIsNotNone(tensor_a.entanglement_id)
        self.assertEqual(tensor_a.entanglement_id, tensor_b.entanglement_id)
        print(f"Nodes Entangled with ID: {tensor_a.entanglement_id}")

        # 3. Modify A
        new_wave = FrequencyWave(frequency=100.0, amplitude=1.0, phase=0.0)
        new_tensor = SoulTensor(wave=new_wave, spin=0.5)
        # Must use update_node_tensor to trigger shared state update
        self.wave_mechanics.update_node_tensor("NodeA", new_tensor)

        # 4. Check B
        tensor_b_updated = self.wave_mechanics.get_node_tensor("NodeB")
        self.assertEqual(tensor_b_updated.wave.frequency, 100.0)
        self.assertEqual(tensor_b_updated.spin, 0.5)
        print("Success: Node B instant update confirmed.")

    def test_photon_emission(self):
        print("\n--- Testing Photon Emission ---")
        payload = FrequencyWave(frequency=50.0, amplitude=10.0, phase=0.0)

        # Emit from A to C
        photon = self.wave_mechanics.emit_photon("NodeA", "NodeC", payload)

        self.assertIsNotNone(photon)
        self.assertEqual(photon.source_id, "NodeA")
        self.assertEqual(photon.target_id, "NodeC")
        print(f"Photon emitted: {photon.id}")

        # Check if Node C absorbed energy (simulated immediate arrival)
        tensor_c = self.wave_mechanics.get_node_tensor("NodeC")
        # Resonance logic: input 10.0 amplitude should increase local energy
        print(f"Node C Amplitude: {tensor_c.wave.amplitude}")
        self.assertGreater(tensor_c.wave.amplitude, 0.0)
        print("Success: Node C absorbed photon energy.")

    def test_crystallization_cycle(self):
        print("\n--- Testing Crystallization (Ice/Fire Cycle) ---")
        # Initialize World
        dna = {"instinct": "survive"}
        world = World(dna, self.wave_mechanics, logger=None)

        # Add cell from NodeC
        world.add_cell("NodeC", properties={"hp": 100})
        cell = world.materialize_cell("NodeC", force_materialize=True)

        # Modify Cell's Soul (Fire State) via Injection (not direct property set)
        # We must change the FRACTAL state, because sync_soul_to_body overwrites tensor.
        print("Injecting high-frequency tone into soul...")
        center = cell.soul.size // 2
        # Inject a massive tone at 999.0Hz
        cell.soul.inject_tone(center, center, amplitude=1000.0, frequency=999.0, phase=0.0)

        # Force sync to verify it picks up the freq
        cell.sync_soul_to_body()
        print(f"Synced Frequency: {cell.tensor.wave.frequency}")
        self.assertAlmostEqual(cell.tensor.wave.frequency, 999.0, delta=1.0)

        # Verify it hasn't touched KG yet
        kg_tensor_before = self.wave_mechanics.get_node_tensor("NodeC")
        self.assertNotEqual(kg_tensor_before.wave.frequency, 999.0)

        # Kill Cell (Trigger Crystallization)
        print("Freezing Cell (Crystallization)...")
        world.crystallize_cell(cell)

        # Check KG (Ice State)
        kg_tensor_after = self.wave_mechanics.get_node_tensor("NodeC")
        print(f"KG Frequency: {kg_tensor_after.wave.frequency}")
        self.assertAlmostEqual(kg_tensor_after.wave.frequency, 999.0, delta=1.0)
        print("Success: State frozen to KG.")

        # Re-Materialize (Thaw)
        print("Thawing Cell (Materialization)...")
        # Clear internal cache to force reload
        if "NodeC" in world.materialized_cells:
            del world.materialized_cells["NodeC"]

        new_cell = world.materialize_cell("NodeC", force_materialize=True)
        self.assertAlmostEqual(new_cell.tensor.wave.frequency, 999.0, delta=1.0)
        print("Success: State restored from KG.")

if __name__ == '__main__':
    unittest.main()
