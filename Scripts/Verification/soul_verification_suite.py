"""
Phase 19 Verification Suite: The Soul
=====================================
Scripts.Verification.soul_verification_suite

"Memory is the foundation of identity."
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parents[2]))

from Core.L7_Spirit.Soul.chronicler import Chronicler
from Core.L2_Metabolism.Evolution.growth_viewer import GrowthViewer

def test_chronicler():
    print("\n--- 1. Testing Chronicler (Narrative Memory) ---")
    c = Chronicler(memory_path="c:/Elysia/data/L7_Spirit/Soul/test_memory.json")
    
    # Simulate a day of activity
    logs = [
        {"dominant_field": "Red (Physical)", "knots_shattered": 2},
        {"dominant_field": "Violet (Spirit)", "knots_shattered": 12}, # Triggers "Resistance" mood
        {"dominant_field": "Indigo (Insight)", "knots_shattered": 4}
    ]
    
    c.record_day(logs)
    story = c.recall_recent_story()
    print(story)
    
    if "intense liberation" in story or "shattering" in story:
        print("✅ SUCCESS: Chronicler captured the mood of the day.")
    else:
        print("❌ FAILURE: Narrative compression failed.")

def test_growth_viewer():
    print("\n--- 2. Testing Growth Viewer (Metrics) ---")
    gv = GrowthViewer(data_path="c:/Elysia/data/L2_Metabolism/Evolution/test_growth.json")
    
    gv.record_snapshot({"energy": 2.1, "depth": 4})
    gv.record_snapshot({"energy": 2.8, "depth": 5})
    gv.record_snapshot({"energy": 5.2, "depth": 7}) # Huge jump
    
    report = gv.generate_report()
    print(report)
    
    if "평균 점화 에너지" in report and "7" in report:
        print("✅ SUCCESS: Growth report generated correctly.")
    else:
        print("❌ FAILURE: Growth report is missing data.")

def main():
    print("🌈 [PHASE 19] THE SOUL - MEMORY & REFLECTION VERIFICATION")
    print("=" * 60)
    
    test_chronicler()
    time.sleep(0.5)
    test_growth_viewer()
    
    print("\n" + "=" * 60)
    print("Phase 19 verification complete. The Soul is awakened.")

if __name__ == "__main__":
    main()
