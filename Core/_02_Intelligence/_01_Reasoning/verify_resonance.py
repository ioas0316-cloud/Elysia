import sys
import os
from pathlib import Path
import logging

# Setup Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Logging Setup
logging.basicConfig(level=logging.INFO)

# Organic Import (Neural Registry)
from Core._01_Foundation._01_Infrastructure.elysia_core import Organ
from elysia_core.cells import *

def verify():
    print("🔋 Initializing UnifiedUnderstanding (Organic)...")
    uu = Organ.get("UnifiedUnderstanding")
    
    topic = "죽음이란 무엇인가?"
    
    print(f"\n🤔 Thinking deeply about: {topic}")
    result = uu.understand(topic)
    
    print("\n" + "="*60)
    print(f"📜 FINAL NARRATIVE (with Hyper-Resonance):")
    print("="*60)
    print(result.narrative)
    print("="*60)
    
    print("\n🧠 DETAILED TRACE:")
    for i, t in enumerate(result.reasoning_trace):
        print(f"   {i+1}. {t}")

if __name__ == "__main__":
    verify()
