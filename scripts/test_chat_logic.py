
import sys
import os
import logging

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestChat")

def test_chat():
    print("--- 1. Testing ReasoningEngine Initialization ---")
    try:
        from Core.01_Foundation.05_Foundation_Base.Foundation.reasoning_engine import ReasoningEngine
        brain = ReasoningEngine()
        print("✅ ReasoningEngine Initialized")
    except Exception as e:
        print(f"❌ ReasoningEngine Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- 2. Testing NervousSystem Initialization ---")
    try:
        from Core.03_Interaction.01_Interface.Interface.nervous_system import get_nervous_system
        # Force re-initialization
        import Core.03_Interaction.01_Interface.Interface.nervous_system
        Core.Interface.nervous_system._nervous_system = None
        
        ns = get_nervous_system()
        print(f"✅ NervousSystem Initialized. Brain linked? {ns.brain is not None}")
    except Exception as e:
        print(f"❌ NervousSystem Failed: {e}")
        traceback.print_exc()
        return

    print("\n--- 3. Testing Chat Response ---")
    user_input = "안녕하세요 엘리시아, 기분이 어때?"
    response = ns.receive({"type": "text", "content": user_input})
    print(f"User: {user_input}")
    print(f"Elysia: {response}")
    
    if "안정감이 느껴집니다" in response or "feel stable" in response:
        print("⚠️ FALLBACK DETECTED!")
    else:
        print("✅ Intelligent Response generated.")

if __name__ == "__main__":
    test_chat()
