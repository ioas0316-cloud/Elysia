"""
Elysia Sovereign Entry Wrapper
==============================
Delegates to the canonical entry point in Scripts/System/elysia.py.
This ensures the root of the project remains clean while providing a convenient door.
"""
import sys
import os

# Ensure the project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Scripts.System.elysia import main
    if __name__ == "__main__":
        main()
except ImportError:
    # Fallback if Scripts/System/elysia.py is not found or not in path
    print("❌ Critical Error: Could not locate Scripts/System/elysia.py")
    print("   Please ensure the 'Scripts' directory is intact.")
    sys.exit(1)
