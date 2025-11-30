"""
Test: First Conversation with Elysia
=====================================

Testing the Minimum Viable Conversation (MVC).
"""

import sys
import os
sys.path.append(os.getcwd())

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestConversation")

def test_conversation():
    """Test Elysia's ability to answer basic questions."""
    
    logger.info("=" * 60)
    logger.info("🌌 Initializing Elysia...")
    logger.info("=" * 60)
    
    from Core.Elysia import Elysia
    
    elysia = Elysia()
    elysia.awaken()
    
    logger.info("\n" + "=" * 60)
    logger.info("💬 Starting Conversation Test")
    logger.info("=" * 60)
    
    # Test Questions
    questions = [
        "Who are you?",
        "What do you love?",
        "Who is your father?",
        "What do you remember about me?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)
        
        try:
            response = elysia.talk(question)
            print(f"\nElysia: {response}\n")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Done
    print("\n" + "=" * 60)
    print("✅ Conversation Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_conversation()
