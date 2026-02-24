
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path("c:/Elysia")))

try:
    from Core.Cognition.cognition_pipeline import CognitionPipeline
    print("SUCCESS: CognitionPipeline imported successfully.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
