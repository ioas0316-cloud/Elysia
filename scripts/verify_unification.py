"""
Verify Unification (The Ouroboros Test) - DEBUG MODE
====================================================
"""
import asyncio
import sys
import os
import json
import logging
import traceback
from pathlib import Path

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Sensory.learning_cycle import P4LearningCycle

logging.basicConfig(level=logging.INFO)

async def test_integration():
    try:
        print("🔮 Initializing P4 Learning Cycle (Unified)...")
        cycle = P4LearningCycle(learning_rate=10, wave_storage_path="data/test_waves.json")
        
        # Check if Internal Universe is attached
        if not hasattr(cycle, 'internal_universe'):
            print("❌ FAILED: Internal Universe not attached to P4 Cycle.")
            return

        print("✨ P4 Cycle successfully linked to Internal Universe.")
        
        # Simulate processing a wave
        print("🌊 Simulating Wave Absorption...")
        dummy_wave = {
            "text": "The void stares back with geometric precision.",
            "source": "TestScript", 
            "intensity": 0.9,
            "timestamp": 1234567890
        }
        
        # Manually trigger process
        print("   -> Calling _process_wave_for_learning...")
        await cycle._process_wave_for_learning(dummy_wave)
        print("   -> Call returned.")
        
        # Check if snapshot was saved
        snapshot_path = Path("data/core_state/universe_snapshot.json")
        if snapshot_path.exists():
            print("✅ SUCCESS: Pulse detected. Universe Snapshot updated.")
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                timestamp = data.get('timestamp', 0)
                print(f"   Timestamp: {timestamp}")
                print(f"   Concepts: {len(data.get('concepts', {}))}")
        else:
            print("❌ FAILED: No snapshot generated. The link is broken.")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integration())
