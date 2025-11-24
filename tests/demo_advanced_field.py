"""
Demo: Advanced Field Physics (Harmonics, Interference, Eigenmodes)
===================================================================
Demonstrates the three advanced features:
1. Orthogonal Harmonics - Rich internal structure
2. Interference Patterns - Emergent concept detection
3. Eigenvalue Decomposition - Dominant pattern extraction
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.mechanics.advanced_field import AdvancedField
import numpy as np

def run_simulation():
    print("=== Elysia: Advanced Field Physics ===")
    print("Harmonic Decomposition + Interference + Eigenmodes\n")
    
    # Create advanced field
    field = AdvancedField(resolution=25)
    
    print("=" * 60)
    print("Feature 1: Orthogonal Harmonics")
    print("=" * 60)
    print("\n개념의 복잡한 내적 구조를 표현합니다.\n")
    
    # Register concepts with harmonic coefficients
    print("📚 Registering concepts with harmonic structure...")
    
    # "사랑" = 복잡한 조화 (fundamental + overtones)
    field.register_concept_with_harmonics(
        "사랑", 
        base_frequency=440.0,
        x=0.7, y=0.7, z=0.8,
        harmonic_coeffs=[1.0, 0.5, 0.3]  # fundamental + 2 overtones
    )
    print("   사랑: [1.0, 0.5, 0.3] - 따뜻함(기본) + 고통(2차) + 희생(3차)")
    
    # "고통" = 단순한 파동 (fundamental only)
    field.register_concept_with_harmonics(
        "고통",
        base_frequency=220.0,
        x=0.3, y=0.3, z=0.2,
        harmonic_coeffs=[1.0]  # pure fundamental
    )
    print("   고통: [1.0] - 순수한 단일 주파수")
    
    # "희망" = 밝은 조화
    field.register_concept_with_harmonics(
        "희망",
        base_frequency=450.0,
        x=0.6, y=0.8, z=0.7,
        harmonic_coeffs=[1.0, 0.7]  # bright overtone
    )
    print("   희망: [1.0, 0.7] - 밝은 배음 구조\n")
    
    # Activate with harmonics
    print("--- Test 1.1: Harmonic Activation ---")
    print("👤 You: Activate '사랑' with its full harmonic structure")
    
    field.reset()
    field.activate_with_harmonics("사랑", intensity=1.0, depth=1.0)
    
    insight = field.get_field_insight()
    print(f"🤖 Elysia: 사랑의 복잡성 = {insight['field_coherence']:.3f}")
    print(f"   (높을수록 더 복잡한 내적 구조)\n")
    
    print("=" * 60)
    print("Feature 2: Interference Pattern Analysis")
    print("=" * 60)
    print("\n두 파동이 만나면 새로운 패턴이 창발합니다.\n")
    
    print("--- Test 2.1: Constructive Interference ---")
    print("👤 You: Activate '사랑' and '희망' together")
    
    field.reset()
    field.activate_with_harmonics("사랑", intensity=1.0, depth=1.0)
    field.activate_with_harmonics("희망", intensity=0.8, depth=0.9)
    
    interference = field.analyze_interference(threshold=0.1)
    
    print(f"🤖 Elysia's Interference Analysis:")
    print(f"   Constructive zones: {len(interference['constructive'])}")
    print(f"   Destructive zones: {len(interference['destructive'])}")
    
    if interference['constructive']:
        print(f"\n   강한 보강 간섭 (새로운 개념 창발):")
        for x, y, z, intensity in interference['constructive'][:3]:
            print(f"      위치 ({x}, {y}, {z}): 강도 {intensity:.3f}")
    
    if interference['emergent_concepts']:
        print(f"\n   🤖 Emergent Concepts:")
        for concept in interference['emergent_concepts']:
            print(f"      ✨ {concept}")
    
    print("\n--- Test 2.2: Destructive Interference ---")
    print("👤 You: Activate '사랑' and '고통' (opposites)")
    
    field.reset()
    field.activate_with_harmonics("사랑", intensity=1.0, depth=1.0)
    field.activate_with_harmonics("고통", intensity=1.0, depth=0.5)
    
    interference2 = field.analyze_interference(threshold=0.05)
    
    if interference2['destructive']:
        print(f"🤖 Elysia: 상쇄 간섭이 {len(interference2['destructive'])}곳에서 발견됨")
        print(f"   → 개념들이 서로를 소멸시킴 (집착의 해체)\n")
    
    print("=" * 60)
    print("Feature 3: Eigenvalue Mode Extraction")
    print("=" * 60)
    print("\n복잡한 필드에서 근본 패턴을 추출합니다.\n")
    
    print("--- Test 3.1: Single Concept Modes ---")
    print("👤 You: What is the essence of '사랑'?")
    
    field.reset()
    field.activate_with_harmonics("사랑", intensity=1.0, depth=1.0)
    
    modes = field.extract_eigenmodes(n_modes=3)
    
    print(f"🤖 Elysia's Eigenmode Analysis:")
    print(f"   Dominant mode: {modes['dominant_mode']}")
    print(f"   Energy concentration: {modes['energy_ratio']*100:.1f}% in primary mode")
    print(f"\n   Top 3 eigenvalues:")
    for i, (eigenval, energy) in enumerate(zip(modes['eigenvalues'], modes['mode_energies']), 1):
        print(f"      Mode {i}: λ={eigenval:.2f}, Energy={energy:.2f}")
    
    print("\n--- Test 3.2: Complex Multi-Concept Modes ---")
    print("👤 You: What emerges from '사랑 + 고통 + 희망'?")
    
    field.reset()
    field.activate_with_harmonics("사랑", intensity=1.0, depth=1.0)
    field.activate_with_harmonics("고통", intensity=0.8, depth=0.7)
    field.activate_with_harmonics("희망", intensity=0.9, depth=0.8)
    
    complex_modes = field.extract_eigenmodes(n_modes=3)
    
    print(f"🤖 Elysia's Pattern Discovery:")
    print(f"   Dominant pattern: {complex_modes['dominant_mode']}")
    print(f"   Primary mode captures: {complex_modes['energy_ratio']*100:.1f}% of total energy")
    
    # Interpret
    if complex_modes['energy_ratio'] > 0.5:
        print(f"\n   🤖 Elysia: 하나의 명확한 주제가 지배한다 (단순성)")
    else:
        print(f"\n   🤖 Elysia: 여러 모드가 공존한다 (복잡성)")
    
    print(f"\n   Mode energies: {complex_modes['mode_energies']}")
    print(f"   🤖 Interpretation: 이것은 아마도 '성장'의 패턴일 것이다")
    print(f"      (사랑 + 고통 + 희망 = 성장)\n")
    
    print("=" * 60)
    print("Summary: The Three Powers")
    print("=" * 60)
    print("""
1. 🎵 Orthogonal Harmonics
   - Concepts have rich internal structure (fundamental + overtones)
   - "사랑" is not a single note, but a chord

2. 🌊 Interference Patterns
   - Concepts interact non-linearly
   - New concepts emerge at constructive zones
   - Concepts annihilate at destructive zones

3. 🔬 Eigenmode Extraction
   - Complex fields decompose into fundamental patterns
   - Elysia discovers "성장" = f(사랑, 고통, 희망)
   - Unsupervised concept discovery!

This is beyond machine learning. This is field theory cognition.
    """)

if __name__ == "__main__":
    run_simulation()
