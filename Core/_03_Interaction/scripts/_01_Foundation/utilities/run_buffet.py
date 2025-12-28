
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation._05_Governance.Foundation.Mind.model_distillery import ModelDistillery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print(f"🍷 Welcome to The Model Buffet")
    
    distillery = ModelDistillery()
    
    # The Menu
    menu = [
        # "tinyllama", # Appetizer (Consumed)
        # "qwen2-0.5b", # Main Course (Consumed)
        "smollm"      # Dessert (Sweet & Light)
    ]
    
    # The Interrogation (Playful Questions for the Dessert)
    questions = [
        "Tell me a joke about a quantum physicist.",
        "What is the best flavor of ice cream for an AI?",
        "If you could fly, where would you go?",
        "Describe the sound of a rainbow.",
        "Who is your favorite fictional robot?"
    ]
    
    try:
        for course in menu:
            print(f"\n🔔 Serving Course: {course}")
            distillery.serve_course(course, questions)
            
    except KeyboardInterrupt:
        print("\n🛑 Buffet interrupted by user.")
    except Exception as e:
        print(f"\n❌ Buffet failed: {e}")
        
    print("\n✨ The Buffet is Closed.")

if __name__ == "__main__":
    main()
