"""Phase 2 verification probe."""
import sys, os
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

dna = SeedForge.forge_soul("Elysia")
monad = SovereignMonad(dna)

print(f"Has goal_generator: {hasattr(monad, 'goal_generator')}")
print(f"Has self_inquiry: {hasattr(monad, 'self_inquiry')}")

# Run enough pulses to generate goals (need 5+ trajectory snapshots, then 20 pulse cooldown)
print("\nRunning 250 pulses...")
for i in range(250):
    monad.pulse(dt=0.1)

print(f"Trajectory size: {monad.trajectory.size}")
print(f"Growth score: {monad.growth_report.get('growth_score', '?')}")
print(f"Growth trend: {monad.growth_report.get('trend', '?')}")

print(f"\nGoal report: {monad.goal_report}")
print(f"Goals generated: {monad.goal_generator.total_generated}")
print(f"Active goals: {monad.goal_generator.active_count}")

inq = monad.self_inquiry.get_status_summary()
print(f"Inquiries: {inq}")

print("\n=== PHASE 2 VERIFICATION COMPLETE ===")
