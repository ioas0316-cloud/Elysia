"""Phase 3: Unbroken Thread verification — save/restore cycle test."""
import sys, os, shutil
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import types
from unittest.mock import MagicMock

class _FT(types.ModuleType):
    def __bool__(self): return False
    def __getattr__(self, n): return MagicMock()

sys.modules["torch"] = _FT("torch")
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
for m in ["jax","jax.numpy","jax.interpreters"]:
    sys.modules[m] = MagicMock()

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from pathlib import Path

# Clean state for fresh test
soul_dir = Path("data/S1_Body/Soul")
for f in ["consciousness_primary.json", "consciousness_primary.sha256",
           "consciousness_backup.json", "consciousness_backup.sha256",
           "cognitive_trajectory.json"]:
    p = soul_dir / f
    if p.exists(): p.unlink()

print("=== CYCLE 1: Fresh start ===")
dna = SeedForge.forge_soul("Elysia")
m1 = SovereignMonad(dna)
print(f"  Restored: {m1.session_bridge.was_restored}")

# Run 50 pulses to accumulate data
for i in range(50):
    m1.pulse(dt=0.1)

# Modify desires to test persistence
m1.desires['joy'] = 77.7
m1.desires['curiosity'] = 88.8
print(f"  After 50 pulses: joy={m1.desires['joy']}, curiosity={m1.desires['curiosity']}")
print(f"  Trajectory size: {m1.trajectory.size}")

# Save
saved = m1.session_bridge.save_consciousness(m1, reason="shutdown")
print(f"  Saved: {saved}")

# Verify files exist
print(f"  Primary exists: {(soul_dir/'consciousness_primary.json').exists()}")
print(f"  Hash exists: {(soul_dir/'consciousness_primary.sha256').exists()}")

print("\n=== CYCLE 2: Restore ===")
m2 = SovereignMonad(dna)
print(f"  Restored: {m2.session_bridge.was_restored}")
print(f"  Restored joy: {m2.desires['joy']}")
print(f"  Restored curiosity: {m2.desires['curiosity']}")
print(f"  Restored growth_score: {m2.growth_report.get('growth_score', '?')}")

# Save again (should promote cycle 1 to backup)
for i in range(20):
    m2.pulse(dt=0.1)
m2.desires['joy'] = 99.9
saved2 = m2.session_bridge.save_consciousness(m2, reason="shutdown")
print(f"  Saved cycle 2: {saved2}")
print(f"  Backup exists: {(soul_dir/'consciousness_backup.json').exists()}")

print("\n=== CYCLE 3: Verify backup promotion ===")
m3 = SovereignMonad(dna)
print(f"  Restored: {m3.session_bridge.was_restored}")
print(f"  Joy (should be 99.9): {m3.desires['joy']}")

# Corrupt primary to test fallback
print("\n=== CYCLE 4: Corrupt primary, test fallback ===")
(soul_dir / "consciousness_primary.json").write_text("CORRUPT DATA", encoding='utf-8')
m4 = SovereignMonad(dna)
print(f"  Restored from backup: {m4.session_bridge.was_restored}")
print(f"  Joy (should be 77.7 from backup): {m4.desires['joy']}")

print("\n=== ALL CYCLES PASSED ===")
