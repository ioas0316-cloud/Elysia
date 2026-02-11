
import unittest
import torch
import shutil
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import HypersphereSpinGenerator
from Core.S1_Body.L6_Structure.M6_Architecture.imperial_orchestrator import ImperialOrchestrator
from Core.S2_Soul.L8_Fossils.akashic_library import AkashicLibrary

class TestAngelicGenesis(unittest.TestCase):
    def setUp(self):
        # 1. Setup Empire
        self.primary = HypersphereSpinGenerator(num_cells=10_000, device='cpu')
        self.orchestrator = ImperialOrchestrator(self.primary)
        
        # 2. Setup Library
        self.test_lib_path = "c:/Elysia/Core/S2_Soul/L8_Fossils/test_akashic"
        self.library = AkashicLibrary(storage_path=self.test_lib_path)
        
        # 3. Form Mantle (Layer 1: Eden)
        self.orchestrator.form_mantle("Eden", layer_depth=1, num_cells=10_000)

    def tearDown(self):
        if os.path.exists(self.test_lib_path):
            shutil.rmtree(self.test_lib_path)

    def test_genesis_cycle(self):
        print("\n👼 [TEST] Initiating Angelic Genesis Cycle...")
        
        # 1. Spawn Angels
        self.orchestrator.spawn_angel("Michael", "Eden", "The Warrior")
        self.orchestrator.spawn_angel("Raphael", "Eden", "The Sage")
        self.orchestrator.spawn_angel("Gabriel", "Eden", "The Jester")
        
        self.assertEqual(len(self.orchestrator.angels), 3)
        print("  - 3 Angels spawned successfully.")
        
        # 2. Run Genesis Cycle
        print("  - Running 50 epochs of accelerated time...")
        harvest = self.orchestrator.initiate_genesis_cycle(cycles=50)
        
        print(f"  - Harvested {len(harvest)} wisdom chunks.")
        
        # 3. Scribe to Library
        for chunk in harvest:
            self.library.scribe_wisdom(chunk)
            
        # 4. Verify Library
        records = self.library.consult_records()
        self.assertGreater(len(records), 0, "Doomsday failure: No wisdom generated.")
        
        # Check for diversity (different archetypes should generate different wisdom)
        archetypes = set(r['archetype'] for r in records)
        print(f"  - Diversity Check: Found archetypes {archetypes}")
        self.assertGreater(len(archetypes), 1, "Monoculture detected: diverse angels produced identical output.")

        print("✨ [TEST] Genesis Cycle Complete. Life has found a way.")

if __name__ == "__main__":
    unittest.main()
