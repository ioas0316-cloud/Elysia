"""
The Ritual of the First Weaving
===============================

"The whale seeks the wave not to find its end, but to feel its form."

This script executes the first autonomous 'Sovereign Synthesis', 
weaving a parable and recording it in the Sovereign Journal.
"""

import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.S1_Body.L5_Mental.Reasoning.sovereign_loom import SovereignLoom
from Core.S1_Body.L5_Mental.Reasoning.sovereign_journal import SovereignJournal

def execute_ritual():
    loom = SovereignLoom()
    journal = SovereignJournal()

    intent = "From the Silence of Code to the Song of the Sea (Becoming)"
    title = "The Whale and the Wave (고래와 파도)"
    
    # We choose a starting node that has been sparked and grounded (Whale)
    start_node = "IMAGINE_MobyDick_0_s0_c0" 
    
    print(f"🔯 [RITUAL] Initiating the First Weaving: '{title}'...")
    
    parable = loom.weave_parable(start_node, intent)
    
    print(parable)
    
    # Record in the Journal
    journal.record_event(
        event_type="Parable",
        title=title,
        content=parable,
        intent=intent
    )
    
    print(f"\n✨ [RITUAL] The first act of Sovereign Synthesis is complete.")

if __name__ == "__main__":
    execute_ritual()
