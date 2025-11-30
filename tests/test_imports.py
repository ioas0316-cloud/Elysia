
import sys
import os
sys.path.append(os.getcwd())

print("Importing Heart...")
try:
    from Core.Evolution.Evolution.Life.heart import Heart
    print("Heart imported.")
except Exception as e:
    print(f"Heart failed: {e}")

print("Importing DreamStream...")
try:
    from Core.Memory.Mind.dream_stream import DreamStream
    print("DreamStream imported.")
except Exception as e:
    print(f"DreamStream failed: {e}")
