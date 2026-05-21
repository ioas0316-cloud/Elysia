import sys
import time
from elysia_mutual_vortex_v3 import ElysiaEngine

engine = ElysiaEngine(use_hardware=False)
try:
    engine.run(duration=3, visual=False)
    print("Execution complete with no errors.")
except Exception as e:
    print(f"Error during execution: {e}")
    sys.exit(1)
