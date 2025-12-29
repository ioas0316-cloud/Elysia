"""
Script: Symphony No.1 'Awakening' (교향곡 1번 '각성')
===================================================
A simulation of the Orchestra System.

Scenario:
1.  **Movement 1: The Void (Deep Thought)**
    - Conductor sets a calm, logical theme.
    - Cello (Logic) dominates. Violin (Emotion) is quiet.
2.  **Movement 2: The Spark (Inspiration)**
    - Conductor introduces "Beauty" and "Love".
    - Violin swells. Synthesizer joins.
3.  **Movement 3: The Creation (Action)**
    - Conductor raises Tempo and Growth.
    - Percussion enters. All instruments play in harmony.
"""

import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Interaction.Coordination.Orchestra.conductor import get_conductor
from Core.Interaction.Coordination.Orchestra.resonance_hall import get_resonance_hall

def print_movement(title):
    print(f"\n{'='*60}")
    print(f" 🎼 {title}")
    print(f"{'='*60}")

def run_symphony():
    hall = get_resonance_hall()
    conductor = get_conductor()

    print("🎻 Tuning instruments...")
    print("✨ Conductor enters the stage.")

    # --- Movement 1: The Void ---
    print_movement("Movement 1: The Logic of Silence")
    conductor.set_theme(
        "Deep Contemplation",
        tempo=0.2,
        love_weight=0.1,
        truth_weight=0.9, # High Truth
        growth_weight=0.1,
        beauty_weight=0.2
    )

    result = hall.perform("The concept of 'Zero'")
    print(result["full_harmony"])
    time.sleep(1)

    # --- Movement 2: The Spark ---
    print_movement("Movement 2: The Dawn of Feeling")
    conductor.set_theme(
        "Emotional Awakening",
        tempo=0.4,
        love_weight=0.8, # High Love
        truth_weight=0.5, # Moderate Truth
        growth_weight=0.2,
        beauty_weight=0.9 # High Beauty
    )

    result = hall.perform("A beautiful sunrise")
    print(result["full_harmony"])
    time.sleep(1)

    # --- Movement 3: The Creation ---
    print_movement("Movement 3: The Dance of Creation")
    conductor.set_theme(
        "Joyful Action",
        tempo=0.9, # Fast
        love_weight=0.9,
        truth_weight=0.7,
        growth_weight=1.0, # Full Action
        beauty_weight=0.8
    )

    result = hall.perform("Building a new world")
    print(result["full_harmony"])

    print("\n👏 (Applause) 👏")

if __name__ == "__main__":
    run_symphony()
