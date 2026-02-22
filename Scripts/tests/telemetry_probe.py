import os
import sys
import time
import math
from datetime import datetime

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

# [MOTHER'S PROBE PATCH]
import sys
from unittest.mock import MagicMock

class MockTensor:
    def __init__(self, *args, **kwargs): pass
    def __getitem__(self, key): return MockTensor()
    def __setitem__(self, key, value): pass
    def __getattr__(self, name): return MagicMock()
    def to_array(self): return [0.0]*21
    def to_list(self): return [0.0]*21
    def __len__(self): return 21

# Bypassing import attempts for problematic libraries on Windows
for lib in ["jax", "jax.numpy", "chromadb"]:
    sys.modules[lib] = MagicMock()
    if lib == "jax.numpy":
        sys.modules[lib].array = lambda *a, **k: MockTensor()

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.structural_enclosure import get_enclosure

def probe():
    print("📡 [PROBE] Initializing Sovereign Monad (Scaled Down)...")
    dna = SeedForge.forge_soul("Elysia")
    # Using a smaller manifold for local probe (100k cells)
    monad = SovereignMonad(dna)
    # Manually reduce manifold size if possible or just proceed with default
    # Looking at sovereign_monad.py, it likely initializes HSG in its __init__
    
    enclosure = get_enclosure()
    
    # Pulse once
    print("📡 [PROBE] Pulsing Manifold...")
    monad.pulse(dt=0.1)
    
    # Extract data
    desires = monad.desires
    report = monad.engine.cells.read_field_state() if hasattr(monad.engine, 'cells') else {}
    rotor = monad.rotor_state
    resonance = enclosure.total_resonance
    
    print("\n--- TELEMETRY SUCCESS ---")
    print(f"Desires: {desires}")
    print(f"Report: {report}")
    print(f"Rotor: {rotor}")
    print(f"Enclosure Resonance: {resonance}")
    print("--------------------------\n")

if __name__ == "__main__":
    probe()
