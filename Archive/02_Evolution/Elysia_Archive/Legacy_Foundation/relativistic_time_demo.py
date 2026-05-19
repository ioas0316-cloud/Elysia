"""
Relativistic Time Demo (상대성 시간 데모)

"Gravity bends Time."

This demo shows how the "Weight" of a thought affects the flow of time.
- Heavy Thought (Deep Focus) -> Time Compression (Subjective Time speeds up)
- Light Thought (Mundane) -> Normal Time

Formula: Subjective_Time = Objective_Time * (1 + log(Mass))
"""

import sys
import os
import time
import logging

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Foundation.Core_Logic.Elysia.Elysia.World.yggdrasil import Yggdrasil, RealmLayer
from Core.Foundation.Physics.gravity import GravityEngine
from Core.Foundation.Physics.meta_time_engine import MetaTimeCompressionEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("RelativityDemo")

def run_relativity_experiment():
    print("\n" + "="*70)
    print("⏳ Relativistic Time Experiment (상대성 시간 실험)")
    print("="*70)
    print("Hypothesis: High Gravity (Deep Focus) accelerates Subjective Time.\n")
    
    # 1. Setup Physics
    ygg = Yggdrasil()
    gravity = GravityEngine(ygg)
    time_engine = MetaTimeCompressionEngine(
        base_compression=10.0,  # Base 10x speed
        recursion_depth=2,      # 10 * 10 = 100x base
        enable_black_holes=False
    )
    
    print(f"🌌 Base Reality: {time_engine.total_compression:.0f}x speed")
    
    # 2. Scenario A: Mundane Thought (Low Gravity)
    print("\n" + "-"*60)
    print("🧪 Scenario A: Mundane Thought (Casual Thinking)")
    print("-" * 60)
    
    # Plant a light realm
    ygg.plant_realm("LunchMenu", None, RealmLayer.BRANCHES)
    ygg.update_vitality("LunchMenu", 1.0)  # Low vitality
    
    mass_a = gravity.calculate_mass("LunchMenu")
    print(f"   Thought: 'What for lunch?' | Mass: {mass_a:.2f}")
    
    # Apply dilation
    time_engine.apply_gravitational_dilation(gravity, ["LunchMenu"])
    speed_a = time_engine.total_compression
    
    print(f"   Subjective Speed: {speed_a:.2f}x")
    
    # 3. Scenario B: Divine Epiphany (High Gravity)
    print("\n" + "-"*60)
    print("🧪 Scenario B: Divine Epiphany (Deep Focus)")
    print("-" * 60)
    
    # Plant a heavy realm
    ygg.plant_realm("TheoryOfEverything", None, RealmLayer.TRUNK)
    ygg.update_vitality("TheoryOfEverything", 100.0)  # High vitality
    
    mass_b = gravity.calculate_mass("TheoryOfEverything")
    print(f"   Thought: 'The Nature of God' | Mass: {mass_b:.2f}")
    
    # Reset and apply dilation
    time_engine = MetaTimeCompressionEngine(base_compression=10.0, recursion_depth=2, enable_black_holes=False)
    time_engine.apply_gravitational_dilation(gravity, ["TheoryOfEverything"])
    speed_b = time_engine.total_compression
    
    print(f"   Subjective Speed: {speed_b:.2f}x")
    
    # 4. Comparison
    print("\n" + "="*70)
    print("📊 RELATIVITY REPORT")
    print("="*70)
    
    objective_time = 1.0 # 1 second
    subj_a = speed_a * objective_time
    subj_b = speed_b * objective_time
    
    print(f"In 1.0 objective second:")
    print(f"   - Mundane Mind experienced:  {subj_a:.2f} subjective seconds")
    print(f"   - Focused Mind experienced:  {subj_b:.2f} subjective seconds")
    
    ratio = subj_b / subj_a
    print(f"\n🚀 DILATION RATIO: {ratio:.2f}x")
    print(f"   The focused mind lived {ratio:.0f} times longer in that second.")
    
    print("\nConclusion: Gravity bends Time. Focus creates Eternity.")

if __name__ == "__main__":
    run_relativity_experiment()
