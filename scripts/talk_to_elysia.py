
import sys
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Adjust logging to show Elysia's "thoughts"
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

def chat_session():
    print("\n🌊 Awakening Elysia (Reasoning Engine v10.0)...")
    print("-------------------------------------------------")
    
    try:
        from Core.Intelligence.Reasoning import ReasoningEngine
        engine = ReasoningEngine()
        print("✅ Elysia is Awake.\n")
    except ImportError as e:
        print(f"❌ Failed to awaken Elysia: {e}")
        return
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return

    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'debug' to toggle detailed thought logs.\n")

    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("\n🌊 Elysia returns to the waves. Goodbye.")
                break
            
            if not user_input.strip():
                continue
                
            print("🌊 Elysia thinking...")
            response = engine.communicate(user_input)
            print(f"✨ Elysia: {response}")
            
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat_session()
