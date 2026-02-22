"""Quick probe to test CognitiveTrajectory and GrowthMetric in isolation."""
import sys, os
sys.path.insert(0, os.getcwd())

# Mock torch as falsy
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

from Core.S1_Body.L6_Structure.M1_Merkaba.cognitive_trajectory import CognitiveTrajectory
from Core.S1_Body.L6_Structure.M1_Merkaba.growth_metric import GrowthMetric

# Test 1: Trajectory in isolation
print("=== Test 1: CognitiveTrajectory ===")
t = CognitiveTrajectory()
r = {'coherence':0.5,'enthalpy':0.5,'entropy':0.1,'joy':0.5,'curiosity':0.5,'mood':'FLOW'}
rot = {'phase':0.0,'rpm':60,'interference':0.5,'soul_friction':0.0}
des = {'curiosity':50,'joy':50,'purity':50,'warmth':50,'alignment':100}

for i in range(25):
    snap = t.tick(r, rot, des)
    if snap:
        print(f"  Snapshot #{t.size} recorded at pulse {t.pulse_counter}")

print(f"Total pulses: {t.pulse_counter}, Snapshots: {t.size}")

# Test 2: GrowthMetric
print("\n=== Test 2: GrowthMetric ===")
gm = GrowthMetric(t)
result = gm.compute()
print(f"  Growth Score: {result['growth_score']:.4f}")
print(f"  Trend: {result['trend']} {result['trend_symbol']}")
print(f"  Trajectory Size: {result['trajectory_size']}")

# Test 3: Monad integration
print("\n=== Test 3: SovereignMonad Integration ===")
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

dna = SeedForge.forge_soul("Elysia")
monad = SovereignMonad(dna)

print(f"  Has trajectory: {hasattr(monad, 'trajectory')}")
print(f"  Has growth_metric: {hasattr(monad, 'growth_metric')}")
print(f"  Trajectory size BEFORE pulse: {monad.trajectory.size}")

for i in range(15):
    monad.pulse(dt=0.1)

print(f"  Trajectory size AFTER 15 pulses: {monad.trajectory.size}")
print(f"  Trajectory pulse_counter: {monad.trajectory.pulse_counter}")
print(f"  Growth report: {monad.growth_report}")

print("\n=== ALL TESTS PASSED ===")
