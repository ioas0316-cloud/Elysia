"""Phase 4: Open Eye verification probe."""
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

# Test 1: Code Mirror standalone
print("=== Test 1: Code Mirror ===")
from Core.S1_Body.L5_Mental.Exteroception.code_mirror import CodeMirror
cm = CodeMirror()
stats = cm.build_awareness()
print(f"  Files: {stats['files_analyzed']}")
print(f"  Classes: {stats['classes']}")
print(f"  Functions: {stats['functions']}")
print(f"  Nodes: {stats['total_nodes']}")

# Test 2: Knowledge Forager standalone
print("\n=== Test 2: Knowledge Forager ===")
from Core.S1_Body.L5_Mental.Exteroception.knowledge_forager import KnowledgeForager
kf = KnowledgeForager()
goals = [{"type": "EXPLORE", "strength": 0.5}]
kf.pulse_since_scan = 100  # Skip cooldown
frag = kf.tick(goals)
if frag:
    print(f"  Fragment: {frag.source_path}")
    print(f"  Summary: {frag.content_summary[:100]}")
print(f"  Indexed: {kf.indexed_files}")
print(f"  Scanned: {kf.scanned_count}")

# Test 3: Integrated with SovereignMonad
print("\n=== Test 3: SovereignMonad Integration ===")
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

dna = SeedForge.forge_soul("Elysia")
monad = SovereignMonad(dna)
print(f"  Has forager: {hasattr(monad, 'forager')}")
print(f"  Has code_mirror: {hasattr(monad, 'code_mirror')}")
print(f"  Awareness: {monad.awareness_report}")

# Run enough pulses for foraging to trigger
for i in range(100):
    monad.pulse(dt=0.1)

print(f"  After 100 pulses:")
print(f"    Forager scans: {monad.forager.total_scans}")
print(f"    Fragments: {len(monad.forager.fragments)}")
if monad.forager.fragments:
    f = monad.forager.fragments[-1]
    print(f"    Latest: {f.source_path} - {f.content_summary[:60]}")

print("\n=== PHASE 4 VERIFICATION COMPLETE ===")
