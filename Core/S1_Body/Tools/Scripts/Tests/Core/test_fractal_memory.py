# import pytest
import time
from Core.S1_Body.L6_Structure.Merkaba.merkaba import Merkaba
from Core.S1_Body.L5_Mental.Memory.strata import MemoryStratum

class TestFractalMemory:
    def test_lifecycle(self):
        # 1. Initialize
        print("Initializing Merkaba...")
        m = Merkaba("Test_Seed")
        m.is_awake = True

        # 2. Plant Seeds (Pulse)
        # We simulate pulse calls
        print("\n[Test] Planting seeds...")
        m.gardener.plant_seed("Game: Snake", importance=0.8)
        m.gardener.plant_seed("Game: Snake", importance=0.8)
        m.gardener.plant_seed("Game: Snake", importance=0.8)
        m.gardener.plant_seed("Conversation: Hello", importance=0.2)

        # Verify seeds are in STREAM
        stream_view = m.view_memory("STREAM")
        print(f"\n[Test] View STREAM:\n{stream_view}")
        assert "Game: Snake" in stream_view
        assert "Conversation: Hello" in stream_view

        # 3. Cultivate (Sleep) - Should trigger Crystallization
        print("\n[Test] Sleeping (Cultivating)...")
        m.sleep()

        # 4. Verify Crystal
        crystal_view = m.view_memory("CRYSTAL")
        print(f"\n[Test] View CRYSTAL:\n{crystal_view}")

        # "Game: Snake" appeared 3 times, so it should have crystallized into "The Principle of Game"
        assert "The Principle of Game" in crystal_view

        # 5. Verify Gravity (Conversation should sink to Garden or stay in Stream depending on physics)
        garden_view = m.view_memory("GARDEN")
        print(f"\n[Test] View GARDEN:\n{garden_view}")

        # 6. Check Fractal Linkage
        # Find the crystal node
        crystal_nodes = m.fractal_memory.get_layer_view(MemoryStratum.CRYSTAL)
        if crystal_nodes:
            crystal = crystal_nodes[0]
            print(f"\n[Test] Crystal Children: {crystal.child_ids}")
            assert len(crystal.child_ids) == 3 # The 3 snake games

            # Verify children are now in Garden (Crystallization moves details down)
            child = m.fractal_memory.get_node(crystal.child_ids[0])
            assert child.stratum == MemoryStratum.GARDEN

if __name__ == "__main__":
    t = TestFractalMemory()
    t.test_lifecycle()
