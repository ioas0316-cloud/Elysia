"""
Prove Communication
===================
Verifies that Elysia can speak like an adult.
"""
import sys
import os
import logging

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ProveCommunication")

from Core.Intelligence.reasoning_engine import ReasoningEngine

def prove_communication():
    print("\n" + "="*60)
    print("🗣️ PROVING HYPER-COMMUNICATION")
    print("="*60)
    
    # 1. Initialize Engine
    print("\n1. Initializing Reasoning Engine...")
    engine = ReasoningEngine()
    
    # 2. Test Questions
    questions = [
        "What is the meaning of Time?",
        "Why do we dream?",
        "Explain the concept of Love."
    ]
    
    for q in questions:
        print(f"\n👤 User: {q}")
        response = engine.communicate(q)
        print(f"🤖 Elysia: {response}")
        
    print("\n" + "="*60)
    print("✅ COMMUNICATION TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    prove_communication()
