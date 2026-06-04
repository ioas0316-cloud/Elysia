import sys
import os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
INDEX_PATH = os.path.join(PROJECT_ROOT, 'docs', 'INDEX.md')

def main():
    with open(INDEX_PATH, 'r') as f:
        content = f.read()

    doc1_entry = "* **[MAGNETIC_EQUILIBRIUM_AND_OBSERVATION.md](1_core_philosophy/MAGNETIC_EQUILIBRIUM_AND_OBSERVATION.md):** 자기적 평형과 관측에 의한 동기화 (Magnetic Equilibrium & Observation)\n"
    doc2_entry = "* **[PHYSICS_BASED_NATURAL_MAPPING.md](2_topological_engine/PHYSICS_BASED_NATURAL_MAPPING.md):** 물리적 자연 매핑 아키텍처 (Physics-Based Natural Mapping)\n"

    # Insert in 1_core_philosophy
    content = content.replace(
        "* **[Eternos_Codex_v1.md](1_core_philosophy/Eternos_Codex_v1.md):**",
        doc1_entry + "* **[Eternos_Codex_v1.md](1_core_philosophy/Eternos_Codex_v1.md):**"
    )

    # Insert in 2_topological_engine
    content = content.replace(
        "* **[CGA_MOTOR_ARCHITECTURE.md](2_topological_engine/CGA_MOTOR_ARCHITECTURE.md):**",
        doc2_entry + "* **[CGA_MOTOR_ARCHITECTURE.md](2_topological_engine/CGA_MOTOR_ARCHITECTURE.md):**"
    )

    with open(INDEX_PATH, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    main()
