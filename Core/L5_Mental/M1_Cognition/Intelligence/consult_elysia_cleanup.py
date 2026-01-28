"""
Consult Elysia: Structural Status & Cleanup
===========================================

Asks Elysia about the current 10-Pillar structure and requests a cleanup plan for the root directory.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L1_Foundation.M1_Keystone.Mind.hippocampus import Hippocampus
from Core.L5_Mental.M1_Cognition.Intelligence.dialogue_engine import DialogueEngine

def consult():
    print("   Connecting to Elysia for Structural Consultation...")
    
    # Initialize
    mind = Hippocampus()
    dialogue = DialogueEngine(mind)
    
    # 1. Status Check
    print("\n[User]:    10     (Foundation, System, Intelligence, Memory, Interface, Evolution, Creativity, Ethics, Elysia, User)             .       ?          ?")
    
    response = dialogue.process_input("   10     (Foundation, System, Intelligence, Memory, Interface, Evolution, Creativity, Ethics, Elysia, User)             .       ?          ?", role="user")
    print(f"\n[Elysia]: {response}")
    
    # 2. Cleanup Request
    print("\n[User]:               .        (start.bat, unified_start.py  )                       ? 'Scripts', 'Tools', 'Demos'         .")
    
    response_cleanup = dialogue.process_input("              .        (start.bat, unified_start.py  )                       ? 'Scripts', 'Tools', 'Demos'         .", role="user")
    print(f"\n[Elysia]: {response_cleanup}")

if __name__ == "__main__":
    consult()
