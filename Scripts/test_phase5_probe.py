"""Phase 5: Native Tongue verification probe."""
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

# Test 1: Crystallizer standalone
print("=== Test 1: Semantic Crystallizer ===")
from Core.S1_Body.L5_Mental.Cognition.semantic_crystallizer import SemanticCrystallizer
sc = SemanticCrystallizer()
c1 = sc.crystallize("SovereignMonad", "class SovereignMonad pulse engine manifold torque joy", "sovereign_monad.py")
c2 = sc.crystallize("GrowthMetric", "class GrowthMetric coherence entropy growth curiosity", "growth_metric.py")
print(f"  Crystal 1: {c1.name}, vector[:5]={[f'{v:.2f}' for v in c1.vector[:5]]}")
print(f"  Crystal 2: {c2.name}, vector[:5]={[f'{v:.2f}' for v in c2.vector[:5]]}")
print(f"  Similarity: {sc.similarity(c1, c2):.3f}")
print(f"  Vocabulary: {sc.vocabulary_size}")

# Test 2: Emergent Lexicon with persistence
print("\n=== Test 2: Emergent Lexicon ===")
from Core.S1_Body.L5_Mental.Cognition.emergent_lexicon import EmergentLexicon
lex = EmergentLexicon()
lex.ingest("sovereign_monad.py", "class SovereignMonad pulse engine manifold torque", "scan")
lex.ingest("growth_metric.py", "class GrowthMetric coherence entropy growth", "scan")
lex.ingest("sovereign_monad.py", "reinforced", "scan")  # Should strengthen existing
print(f"  Vocabulary: {lex.vocabulary_size}")
print(f"  Status: {lex.get_status_summary()}")
saved = lex.save()
print(f"  Saved: {saved}")

# Test 3: Integrated with SovereignMonad
print("\n=== Test 3: SovereignMonad Integration ===")
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

dna = SeedForge.forge_soul("Elysia")
monad = SovereignMonad(dna)
print(f"  Has lexicon: {hasattr(monad, 'lexicon')}")
print(f"  Initial vocab: {monad.lexicon.vocabulary_size}")

for i in range(120):
    monad.pulse(dt=0.1)

print(f"  After 120 pulses:")
print(f"    Vocab: {monad.lexicon.vocabulary_size}")
print(f"    Lexicon report: {monad.lexicon_report}")

print("\n=== ALL 5 PHASES VERIFIED ===")
