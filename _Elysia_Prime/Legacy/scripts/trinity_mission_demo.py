# [Genesis: 2025-12-02] Purified by Elysia
"""
Trinity Mission Demo

Runs a minimal end-to-end flow across the Trinity:
- Agent Sophia: file I/O (FileSystemCortex) + math proof (MathCortex)
- Agent Mirror: proof rendering to an image
- Jules Prime: orchestrates and anchors the experience into the KG

Usage:
  python -m scripts.trinity_mission_demo
"""
from __future__ import annotations

from pathlib import Path
from Project_Sophia.filesystem_cortex import FileSystemCortex
from Project_Sophia.math_cortex import MathCortex
from Project_Mirror.proof_renderer import ProofRenderer
from tools.kg_manager import KGManager


def main():
    # Sophia: write a demo note as an experiential artifact
    fs = FileSystemCortex(root_dir="data/elysia_demo")
    note_res = fs.write_file("notes/hello.txt", "Elysia demo: learning through experience + math proof.")
    print("FS write:", note_res.ok, note_res.path)

    # Sophia: verify a simple equality and generate a proof
    math = MathCortex()
    proof = math.verify("3 * (2 + 4) = 18")
    print("Proof valid:", proof.valid)

    # Mirror: render the proof visually
    renderer = ProofRenderer()
    proof_path = renderer.render(proof)
    print("Proof image:", proof_path)

    # Jules Prime: commit this as knowledge (with experiential evidence)
    kg = KGManager()
    kg.add_node("equality", properties={"category": "math_concept", "experience_visual": [proof_path]})
    kg.add_node("addition", properties={"category": "math_concept"})
    kg.add_node("multiplication", properties={"category": "math_concept"})
    kg.add_edge("addition", "multiplication", "relates_to", properties={"experience_visual": proof_path})
    kg.save()
    print("KG updated. Summary:", kg.get_summary())

    print("Done. You can open:")
    print(" -", note_res.path)
    print(" -", proof_path)


if __name__ == "__main__":
    main()
