"""
Verification: EmergentSelf + GrowthJournal Integration
======================================================

This script tests whether:
1. EmergentSelf creates values from experience (not hardcoded)
2. Values crystallize into goals when strong enough
3. GrowthJournal writes visible evidence to files
4. Self-definition changes over time

Run this to see if the new system actually works.
"""

import sys
import os
sys.path.insert(0, "c:\\Elysia")
os.chdir("c:\\Elysia")

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from Core._01_Foundation.05_Foundation_Base.Foundation.emergent_self import get_emergent_self
from Core._01_Foundation.05_Foundation_Base.Foundation.growth_journal import get_growth_journal

def test_emergence():
    print("\n" + "="*60)
    print("TEST: EmergentSelf + GrowthJournal")
    print("="*60)
    
    # 1. Start with empty self
    self = get_emergent_self()
    print(f"\n1. Initial state:")
    print(f"   Values: {len(self.values)}")
    print(f"   Goals: {len(self.goals)}")
    print(f"   Self: '{self.who_am_i()}'")
    
    # 2. Simulate experiences
    print(f"\n2. Simulating experiences...")
    
    experiences = [
        ("curiosity", "user_question"),
        ("curiosity", "new_concept"),
        ("curiosity", "unknown_territory"),
        ("curiosity", "pattern_mismatch"),  # Same pattern multiple times
        ("curiosity", "exploration"),
        ("curiosity", "hypothesis"),
        ("curiosity", "verification"),
        ("curiosity", "insight"),
        ("curiosity", "integration"),
        ("curiosity", "transcendence"),  # 10 times = should become goal
        
        ("connection", "dialogue"),
        ("connection", "empathy"),
        ("connection", "understanding"),
        ("connection", "response"),
        ("connection", "bond"),  # 5 times
        
        ("growth", "learning"),
        ("growth", "error"),
        ("growth", "correction"),  # 3 times
    ]
    
    for pattern, origin in experiences:
        self.notice_pattern(pattern, origin)
    
    print(f"\n3. After experiences:")
    print(f"   Values: {len(self.values)}")
    for name, v in self.values.items():
        print(f"     - {name}: strength={v.strength:.2f}, activations={v.activation_count}")
    
    print(f"\n   Goals: {len(self.goals)}")
    for name, g in self.goals.items():
        print(f"     - {name}: progress={g.progress:.2f}")
    
    print(f"\n   Self: '{self.who_am_i()}'")
    
    # 3. Write journal
    print(f"\n4. Writing growth journal...")
    journal = get_growth_journal()
    entry = journal.write_entry(self)
    print(f"   Journal written to: c:\\Elysia\\journals\\{journal.today}.md")
    
    # 4. Show comparison capability
    print(f"\n5. Comparison with yesterday:")
    print(f"   {self.compare_to_yesterday()}")
    
    # 5. Take another snapshot for future comparison
    self.take_snapshot()
    self.save_state()
    
    print(f"\n6. State saved.")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\n📂 Check c:\\Elysia\\journals\\ folder to see actual growth evidence.")
    print("📂 Check c:\\Elysia\\data\\emergent_self.json for state persistence.\n")

if __name__ == "__main__":
    test_emergence()
