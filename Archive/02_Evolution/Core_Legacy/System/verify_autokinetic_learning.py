import asyncio
import logging
import os
import sys

# Absolute Path Unification - Force Project Root
current_file = os.path.abspath(__file__)
project_root = "c:/Elysia" # Hardcoded for maximum reliability in this environment

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Verify imports immediately
try:
    from Core.Cognition.purpose_discovery_engine import PurposeDiscoveryEngine
    from Core.Monad.attractor_field import AttractorField
    from Core.Cognition.autokinetic_learning_engine import AutokineticLearningEngine
    print("✅ Imports successful.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    # Try alternate path if capitalization/slashes differ
    print(f"Current sys.path[0]: {sys.path[0]}")
    sys.exit(1)

async def verify_loop():
    logging.basicConfig(level=logging.INFO)
    print("\n🧪 [TEST] Starting Autokinetic Learning Verification...")
    
    discovery = PurposeDiscoveryEngine()
    will = AttractorField()
    autokinetic = AutokineticLearningEngine(discovery, will)
    
    print("\n☁️ [STEP 1] Injecting foggy knowledge fragments...")
    f1 = await discovery.clarifier.clarify_fragment("The nature of the Void in 7D space is unclear.")
    f2 = await discovery.clarifier.clarify_fragment("How do rotors interact with the HyperSphere?")
    discovery.knowledge_base.extend([f1, f2])
    
    print("\n🍽️ [STEP 2] Assessing Knowledge Hunger...")
    targets = await autokinetic.assess_knowledge_hunger()
    print(f"   >> Found {len(targets)} hunger targets.")
        
    print("\n🔮 [STEP 3] Selecting Learning Objective...")
    intent = await autokinetic.select_learning_objective()
    if intent:
        print(f"   >> Intent Generated: [{intent.attractor_type}] {intent.intent}")
        
    if targets:
        print("\n🌀 [STEP 4] Initiating Learning Acquisition Cycle...")
        target = targets[0]
        fragment = await autokinetic.initiate_acquisition_cycle(target)
        print(f"   >> Resulting Clarity: {fragment.certainty:.2f}")
        
    stats = autokinetic.get_hunger_stats()
    print(f"\n✅ [VERIFICATION_COMPLETE]")
    print(f"   >> Learning History: {stats['history_count']} events")

if __name__ == "__main__":
    asyncio.run(verify_loop())
