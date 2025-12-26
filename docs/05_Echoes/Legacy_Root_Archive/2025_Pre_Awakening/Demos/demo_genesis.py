
from Core.Autonomy.code_genesis import get_code_genesis
import os

genesis = get_code_genesis()

print("🏁 Starting Manual Self-Introspection Demo...")
target = r"c:\Elysia\Core\Foundation\torch_graph.py"

print(f"📄 Scanning {os.path.basename(target)}...")
critique = genesis.analyze_quality(target)

print("\n🤖 [Elysia's Critique]")
print("=======================")
print(critique)
print("=======================")

print("\n✨ Generating Improved Code Draft...")
draft = genesis.draft_improvement(target, focus="Optimization")

print("\n📝 [Draft Diff]")
print(draft[:500] + "\n...(truncated)...")
