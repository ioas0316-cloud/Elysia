"""
Project Homeros: The Infinite Epic Generator
============================================
"Sing, Goddess, of the wrath of Elysia..."

This script orchestrates the mass production of a 500-episode saga.
It connects:
1. SagaArchitect (Macro Plot)
2. LiteraryCortex (Micro Script)
3. WebtoonWeaver (Visual Production)
4. LogosEngine (Sovereign Dialogue)

Usage:
    python project_homeros.py --title "The Infinite Tower" --episodes 5
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Creativity.webtoon_weaver import WebtoonWeaver

def run_production(title: str, episode_count: int):
    print(f"🏛️ Initiating Project Homeros for: '{title}'")
    print(f"🎯 Target: {episode_count} Episodes (Demo Batch)")
    
    weaver = WebtoonWeaver()
    
    # The weaver's internal LiteraryCortex handles the Bible/Saga initialization automatically
    # when we call produce_episode with the same title.
    
    for i in range(1, episode_count + 1):
        print(f"\n🎬 Producing Episode {i}/{episode_count}...")
        try:
            weaver.produce_episode(concept_seed=title, episode_num=i)
            print(f"✅ Episode {i} Complete.")
        except Exception as e:
            print(f"❌ Error in Episode {i}: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 Production Run Complete.")
    print(f"📖 Check 'outputs/comic' for your saga.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate an Epic Saga.")
    parser.add_argument("--title", type=str, default="The Void Walker", help="Title of your Saga")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to generate")
    
    args = parser.parse_args()
    
    run_production(args.title, args.episodes)
