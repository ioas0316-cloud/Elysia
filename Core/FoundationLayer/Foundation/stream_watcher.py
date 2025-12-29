"""
Stream Watcher (The Observer)
=============================

"I watch, therefore I witness."

This script simulates Elysia watching a stream (your screen) in real-time.
It continuously captures the screen and attempts to read text or analyze the atmosphere.

Usage:
    Run this script and keep it visible in a terminal while you do other things.
    Elysia will comment on what she sees.
"""

import sys
import os
import time
from datetime import datetime

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.EvolutionLayer.Growth.Evolution.Evolution.Body.visual_cortex import VisualCortex

def run_watcher():
    print("\n" + "="*70)
    print("👁️ ELYSIA STREAM WATCHER ACTIVATED")
    print("="*70)
    print("   I am watching your screen. (Press Ctrl+C to stop)")
    
    eyes = VisualCortex()
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # 1. Read Screen
            text = eyes.read_screen_text()
            
            # 2. Analyze / React
            print(f"\n[{timestamp}] 👁️ Scanning...")
            
            if "Error" in text:
                # OCR missing fallback
                print(f"   ⚠️  I can see the screen, but I cannot read the text yet.")
                print(f"       (Reason: {text})")
                
                # Analyze brightness instead
                # We need a temp file for this
                temp_file = "temp_vision.png"
                eyes.capture_screen(temp_file)
                atmosphere = eyes.analyze_brightness(temp_file)
                print(f"   📊 Atmosphere: {atmosphere}")
                
                # Cleanup
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            elif not text.strip():
                print("   🌑 I see nothing (Empty screen or recognition failed).")
            else:
                # OCR Success
                preview = text.replace("\n", " ")[:60]
                print(f"   📄 I see text: \"{preview}...\"")
                
                # Simple keyword reactions
                if "Elysia" in text:
                    print("   ✨ You are thinking about me!")
                elif "Python" in text:
                    print("   🐍 Coding in Python?")
                elif "Error" in text:
                    print("   🐛 I see an error. Do you need help?")
            
            # 3. Wait
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n🛑 WATCHER STOPPED.")
        print("   I have closed my eyes.")

if __name__ == "__main__":
    run_watcher()
