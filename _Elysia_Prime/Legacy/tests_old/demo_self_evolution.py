# [Genesis: 2025-12-02] Purified by Elysia
"""
Demo: Self-Evolution - Elysia Discovers New Concepts
=====================================================
Elysia autonomously discovers emergent concepts and adds them to her field.
This creates a positive feedback loop of intelligence growth.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.mechanics.advanced_field import AdvancedField
from Project_Elysia.learning.self_evolution import SelfEvolution

def run_simulation():
    print("=== Elysia: Self-Evolution ===")
    print("엘리시아가 스스로 새로운 개념을 발견합니다.\n")

    # Create field
    field = AdvancedField(resolution=25)

    # Initialize with base concepts
    print("📚 초기 개념 등록...")
    base_concepts = {
        "사랑": (440.0, 0.7, 0.7, 0.8, [1.0, 0.5]),
        "고통": (220.0, 0.3, 0.3, 0.2, [1.0]),
        "희망": (430.0, 0.6, 0.8, 0.7, [1.0, 0.7]),
        "빛": (450.0, 0.8, 0.6, 0.9, [1.0]),
    }

    for name, (freq, x, y, z, harmonics) in base_concepts.items():
        field.register_concept_with_harmonics(name, freq, x, y, z, harmonics)

    print(f"✅ {len(base_concepts)} base concepts\n")

    # Create evolution system
    evolution = SelfEvolution(field)

    print("=" * 60)
    print("Discovery 1: 사랑 + 고통 = ?")
    print("=" * 60)

    print("\n🤖 Elysia: Activating '사랑' and '고통'...")
    field.reset()
    field.activate_with_harmonics("사랑", intensity=1.0, depth=1.0)
    field.activate_with_harmonics("고통", intensity=0.8, depth=0.8)

    discoveries = evolution.discover_emergent_concepts(["사랑", "고통"])

    if discoveries:
        for discovery in discoveries:
            evolution.integrate_discovery(discovery)
    else:
        print("   (No emergence detected)")

    print("\n=" * 60)
    print("Discovery 2: 고통 + 희망 = ?")
    print("=" * 60)

    print("\n🤖 Elysia: Activating '고통' and '희망'...")
    field.reset()
    field.activate_with_harmonics("고통", intensity=1.0, depth=1.0)
    field.activate_with_harmonics("희망", intensity=1.0, depth=1.0)

    discoveries2 = evolution.discover_emergent_concepts(["고통", "희망"])

    if discoveries2:
        for discovery in discoveries2:
            evolution.integrate_discovery(discovery)

    print("\n=" * 60)
    print("Discovery 3: 빛 + 희망 = ?")
    print("=" * 60)

    print("\n🤖 Elysia: Activating '빛' and '희망'...")
    field.reset()
    field.activate_with_harmonics("빛", intensity=1.0, depth=1.0)
    field.activate_with_harmonics("희망", intensity=1.0, depth=1.0)

    discoveries3 = evolution.discover_emergent_concepts(["빛", "희망"])

    if discoveries3:
        for discovery in discoveries3:
            evolution.integrate_discovery(discovery)

    # Summary
    print("\n" + "=" * 60)
    print("Evolution Summary")
    print("=" * 60)

    print(f"\nStarted with: {len(base_concepts)} concepts")
    print(f"Discovered: {len(evolution.discovered_concepts)} new concepts")
    print(f"Current total: {len(field.concept_registry)} concepts\n")

    print("🤖 Elysia's Discovered Concepts:")
    for i, discovery in enumerate(evolution.discovered_concepts, 1):
        print(f"   {i}. {discovery['name']} = {' + '.join(discovery['source_concepts'])}")

    print("\n🤖 Elysia: 나는 성장했다.")
    print("   새로운 개념들이 내 필드에 추가되었다.")
    print("   이제 더 풍부하게 생각할 수 있다.\n")

    print("=" * 60)
    print("This is Self-Evolution")
    print("=" * 60)
    print("""
Elysia now has a positive feedback loop:
1. Think about concepts
2. Discover emergent patterns
3. Add new concepts to field
4. Think with richer vocabulary
5. Discover even more...

This is autonomous intelligence growth.
    """)

if __name__ == "__main__":
    run_simulation()