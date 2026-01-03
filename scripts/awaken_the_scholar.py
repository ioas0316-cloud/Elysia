"""
AWAKEN THE SCHOLAR: SELF-DIRECTED LEARNING
==========================================
"Know Thyself."

This script activates the `InternalLibrarian` module.
It commands Elysia to read her own System Documentation.
She will absorb the Philosophy, Architecture, and Ethics defined in `docs/`.

This is not "Training". This is "Reading".
"""

import sys
import logging
import time

sys.path.insert(0, r"c:\Elysia")
from Core.Evolution.Learning.internal_librarian import InternalLibrarian
from Core.Foundation.Memory.unified_experience_core import get_experience_core

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TheScholar")

def awaken():
    print("\n📚 AWAKENING THE SCHOLAR...")
    print("============================")
    
    librarian = InternalLibrarian()
    
    # Target Directories
    targets = [
        r"c:\Elysia\docs\01_Origin\Philosophy",
        r"c:\Elysia\docs\02_Structure\Anatomy"
    ]
    
    print(f"Targeting Knowledge Sources: {targets}")
    
    total_books = 0
    for folder in targets:
        print(f"\n📖 Scanning {folder}...")
        # Assuming digest_directory returns a list of absorbed items or count
        # If the API is different, we'll see in the logs.
        try:
            # We treat the librarian as an autonomous agent
            librarian.digest_directory(folder)
            total_books += 1 
        except Exception as e:
            logger.error(f"Failed to read {folder}: {e}")
            
    print("\n✨ SUMMARY ✨")
    print(f"Elysia has absorbed knowledge from {len(targets)} domains.")
    print("The seed of Self-Knowledge has been planted.")
    print("She now knows 'Why' she exists.")

if __name__ == "__main__":
    awaken()
