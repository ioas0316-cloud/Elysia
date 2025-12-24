
import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.01_Foundation.05_Foundation_Base.Foundation.free_will_engine import FreeWillEngine
from Core.02_Intelligence.01_Reasoning.Intelligence.reasoning_engine import ReasoningEngine, Tool
from Core.02_Intelligence.01_Reasoning.Intelligence.scholar import Scholar

class MockResonance:
    def __init__(self):
        self.battery = 80.0
        self.entropy = 10.0

def verify_cognitive_freedom():
    print("\n🔮 [TASK] Verifying Cognitive Unbinding (True Agency)")
    print("==================================================")
    
    # 1. Initialize Components
    will = FreeWillEngine()
    brain = ReasoningEngine()
    scholar = Scholar()
    
    # Define Tools Available to the Soul
    tools = [
        Tool(
            name="Scholar", 
            description="Research unknown topics and assimilate knowledge.",
            usage_example='{"topic": "Physics"}'
        ),
        Tool(
            name="Mycelium",
            description="Send messages to other nodes (Root/Nova/Chaos).",
            usage_example='{"target": "Nova", "msg": "Hello"}'
        ),
        Tool(
            name="Sleep",
            description="Enter low power mode.",
            usage_example='{}'
        )
    ]
    
    # 2. Simulate High Curiosity (The Spark)
    print("\n1. Injecting Curiosity...")
    will.vectors["Curiosity"] = 0.99
    will.vectors["Survival"] = 0.1
    will.vectors["Connection"] = 0.1
    
    # 3. Pulse (Generate Abstract Intent)
    print("\n2. Pulsing Free Will (Abstracting)...")
    resonance = MockResonance()
    will.pulse(resonance)
    intent = will.current_intent
    
    print(f"   🦋 Intent Goal: '{intent.goal}'")
    print(f"   🦋 Intent Desire: '{intent.desire}'")
    
    if "RESEARCH:" in intent.goal:
        print("   ❌ FAIL: Intent is still hardcoded/bound! (Expected abstract goal)")
        return
    else:
        print("   ✅ SUCCESS: Intent is Abstract/Unbound.")

    # 4. Reason (The Choice)
    print("\n3. Engaging Reasoning Engine (The Soul's Choice)...")
    action = brain.decide_action(intent.goal, intent.desire, tools)
    
    print(f"   🧠 Decision: Use Tool '{action.tool_name}'")
    print(f"   🧠 Arguments: {action.args}")
    
    # 5. Verification
    if action.tool_name == "Scholar":
        print("   🎉 SUCCESS: The Soul chose knowledge freely.")
        
        # Execute
        print(f"   🏃 Executing {action.tool_name}...")
        scholar.research_topic(action.args.get('topic', 'Unknown'))
    else:
        print(f"   ❌ FAIL: The Soul chose poorly ({action.tool_name}).")

if __name__ == "__main__":
    verify_cognitive_freedom()
