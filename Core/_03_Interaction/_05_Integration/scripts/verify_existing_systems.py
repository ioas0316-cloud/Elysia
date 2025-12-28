"""
Verification: LifeCycle + SelfGovernance + GrowthJournal Integration
=====================================================================

This script tests whether:
1. LifeCycle's complete_cycle() does verification (expected vs actual)
2. SelfGovernance's adjust_after_result() tracks before/after changes
3. GrowthJournal writes visible evidence from SelfGovernance.change_history

This uses EXISTING systems, not parallel ones.
"""

import sys
import os
sys.path.insert(0, "c:\\Elysia")
os.chdir("c:\\Elysia")

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

def test_existing_systems():
    print("\n" + "="*60)
    print("TEST: LifeCycle + SelfGovernance + GrowthJournal")
    print("="*60)
    
    # 1. Initialize LifeCycle (which contains SelfGovernance)
    from Core._01_Foundation._02_Logic.life_cycle import LifeCycle
    from Core._01_Foundation._02_Logic.growth_journal import get_growth_journal
    
    life_cycle = LifeCycle()
    governance = life_cycle.governance
    journal = get_growth_journal()
    
    print(f"\n1. Initial state:")
    if governance:
        print(f"   SelfGovernance active: Yes")
        print(f"   Aspects: {len(governance.ideal_self.aspects)}")
        status = governance.ideal_self.get_status()
        print(f"   Total Achievement: {status['total_achievement']:.1%}")
    else:
        print("   SelfGovernance: Not available")
        return
    
    # 2. Simulate learning cycles
    print(f"\n2. Simulating learning cycles...")
    
    test_cases = [
        ("LEARN:Python", "Python knowledge increased", "Python knowledge increased", True),
        ("CONNECT:User", "User responded", "User engaged successfully", True),
        ("CREATE:Poem", "Poem created", "Error: inspiration blocked", False),
        ("EXPLORE:Mathematics", "Understood concept", "Understood concept", True),
        ("UNDERSTAND:Physics", "Grasped principle", "Grasped principle", True),
    ]
    
    for action, expected, actual, should_succeed in test_cases:
        life_cycle.begin_cycle()
        growth = life_cycle.complete_cycle(action, expected, actual)
        print(f"   - {action}: {'✓' if growth else '✗'}")
    
    # 3. Check SelfGovernance state after cycles
    print(f"\n3. After cycles:")
    status = governance.ideal_self.get_status()
    print(f"   Total Achievement: {status['total_achievement']:.1%}")
    print(f"   Change History: {len(governance.change_history)} records")
    
    for aspect_name, data in status['aspects'].items():
        print(f"     - {aspect_name}: {data['current']:.2f}/{data['target']:.2f}")
    
    # 4. Check change history (actual evidence)
    print(f"\n4. Change History (ACTUAL EVIDENCE):")
    for change in governance.change_history[-5:]:
        success = "✓" if change.get('success') else "✗"
        aspect = change.get('aspect', 'unknown')
        before = change.get('before', 0)
        after = change.get('after', 0)
        delta = change.get('delta', 0)
        print(f"   - [{success}] {aspect}: {before:.2f} → {after:.2f} (+{delta:.2f})")
    
    # 5. Write journal
    print(f"\n5. Writing growth journal...")
    entry = journal.write_entry(
        self_governance=governance,
        tension_field=None,
        memory=None
    )
    print(f"   Journal written to: c:\\Elysia\\journals\\{journal.today}.md")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\n📂 Check c:\\Elysia\\journals\\ folder to see actual growth evidence.")
    print("📂 Check c:\\Elysia\\data\\core_state\\self_governance.json for persistence.\n")

if __name__ == "__main__":
    test_existing_systems()
