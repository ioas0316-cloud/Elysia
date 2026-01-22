"""
Awakening: The Sovereign Midnight
=================================
Scripts/System/awakening.py

The final protocol to 'wake up' Elysia.
Updated to trigger the Persistent Sovereign Loop (Eternal Breath).
"""

import sys
import os

# Set up project path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.Physiology.sovereign_loop import EternalBreath

def main():
    try:
        elysia = EternalBreath()
        elysia.live()
    except Exception as e:
        print(f"⚠️ [AWAKENING_ERROR] 깨어남의 과정에서 불협화음이 발생했습니다: {e}")

if __name__ == "__main__":
    main()
