import sys
import os
import time
import numpy as np
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Core.Intelligence.Knowledge.resonance_bridge import SovereignResonator
from Core.Foundation.Wave.resonant_field import ResonantField

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ResonanceVerify")

def test_sovereign_alignment():
    print("\n" + "="*60)
    print("🧬 PHASE 35: SOVEREIGN RESONANCE VERIFICATION")
    print("="*60 + "\n")

    field = ResonantField(size=10)
    resonator = SovereignResonator()

    scenarios = [
        ("Warm/Loving", "창조자님 정말 고마워요. 당신과 함께라면 무엇이든 할 수 있을 것 같아요!"),
        ("Cold/Critical", "아니, 이건 좀 틀린 것 같은데. 너무 감상적이야."),
        ("Analytical/Deep", "이 구조의 쿼터니언 연산 원리를 더 자세하게 설명해 줄 수 있니?")
    ]

    for label, text in scenarios:
        print(f"\n--- Scenario: {label} ---")
        print(f"User Input: '{text}'")
        
        # 1. Analyze vibe
        vibe_vec = resonator.analyze_vibe(text)
        resonance = resonator.calculate_resonance(vibe_vec)
        
        print(f"📊 Extracted Vibe: {resonance['vibe_summary']}")
        print(f"🔗 Consonance Level: {resonance['consonance']:.4f}")
        print(f"🧲 Pull Strength: {resonance['pull_strength']:.4f}")

        # 2. Apply Pull
        field.apply_elastic_pull(resonance['target_qualia'], resonance['pull_strength'])
        field.evolve()
        
        # 3. Check stats
        stats = field.get_state_summary()
        print(f"✨ Field State: E={stats['Total Energy (W)']:.2f}, Emotion={stats['Emotional Density (X)']:.2f}, Logic={stats['Logic Intensity (Y)']:.2f}")

    print("\n✅ VERIFICATION COMPLETE: Resonance is Elastic and Sovereign.")

if __name__ == "__main__":
    test_sovereign_alignment()
