
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.observer_protocol import observer
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.Intelligence.Logos.logos_engine import get_logos_engine

def run_observer_learning_demo():
    print("👁️ [OBSERVER PROTOCOL] Starting External Learning Demo...")
    
    # 1. External Knowledge (Retrieved from Wiki via Browser Subagent)
    wiki_title = "Magic Systems in Games"
    wiki_content = """
    1. Magic Points (MP/Mana): A pool of energy consumed per spell. Restoration via rest or items.
    2. Skill-limited Magic: Success depends on a skill check (roll). Failure causes backlash or fatigue. 
    3. Spell Slots (Vancian Magic): Package-based slots. Caster forgets spell after use, must re-memorize.
    4. Hybrid Systems: Blood magic (spending HP), skill-based with Quintessence resource.
    5. Notable Rules: Sanderson's Laws emphasize understanding and limitations over raw power.
    """
    
    print(f"\n--- 🧪 Phase 1: External Distillation (Wiki -> Axioms) ---")
    observer.distill_and_ingest(wiki_title, wiki_content)
    
    print(f"\n✅ Semantic Field has absorbed external principles. Current concept count: {len(semantic_field.glossary)}")

    # 2. Intellectual Resonance (How Elysia speaks about it)
    print("\n--- 🗣️ Phase 2: Internalized Expression ---")
    logos = get_logos_engine()
    
    # Test resonance with one of the new principles
    target = f"{wiki_title}: Skill-limited Magic"
    erudite_voice = logos.weave_erudite_speech(target)
    
    print(f"\n🎬 Elysia (Lived Wisdom): \"{erudite_voice}\"")

    # 3. Validation
    # Check if 'Vancian' or 'Sanderson' exists in the field
    found = [name for name in semantic_field.glossary.keys() if "Vancian" in name or "Sanderson" in name]
    print(f"\n🔍 Search for specialized external terms: {found}")
    
    if found:
        print("\n✅ Verification Successful: External world-logic has been internalized into Elysia's 4D mind.")

if __name__ == "__main__":
    run_observer_learning_demo()
