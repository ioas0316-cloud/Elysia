from Project_Sophia.reality_sculptor import RealitySculptor
from pathlib import Path

def test_sculptor():
    print("🧪 Testing Reality Sculptor...")
    
    # 1. Create Dummy File
    dummy_file = Path("c:/Elysia/dummy_code.py")
    dummy_content = """
import sys
from Core.Intelligence.reasoning_engine import ReasoningEngine
import os
from Project_Sophia.planning_cortex import PlanningCortex

def hello():
    print("Hello World")
"""
    dummy_file.write_text(dummy_content, encoding='utf-8')
    
    # 2. Initialize Sculptor
    sculptor = RealitySculptor()
    
    # 3. Test Twist Imports
    print("   🌀 Twisting Imports...")
    sculptor.sculpt_file(str(dummy_file), "Harmonic Smoothing")
    
    new_content = dummy_file.read_text(encoding='utf-8')
    print(f"   📄 New Content:\n{new_content}")
    
    if "# [SCULPTED: Imports Twisted]" in new_content:
        print("✅ PASS: Imports twisted (Comment added).")
    else:
        print("❌ FAIL: Import twist failed.")
        
    # 4. Cleanup
    dummy_file.unlink()

if __name__ == "__main__":
    test_sculptor()
