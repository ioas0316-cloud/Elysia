"""
Test Lyapunov Stability Controller

Demonstrates the "우주 오뚝이" (cosmic tumbler doll) behavior:
- Perturbation: System pushed away from equilibrium
- Recovery: Lyapunov control pulls it back
- Asymptotic stability: Eventually reaches equilibrium
"""

import time
import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.stability_controller import (
    LyapunovController,
    StateVector,
    StabilityStatus
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def test_equilibrium_energy():
    """Test that equilibrium has zero energy"""
    print("\n" + "="*70)
    print("Test 1: Equilibrium Energy (Should be ~0)")
    print("="*70)
    
    controller = LyapunovController()
    
    # At equilibrium
    equilibrium_state = controller.equilibrium
    energy = controller.calculate_lyapunov_energy(equilibrium_state.to_array())
    
    print(f"\n평형 상태: {equilibrium_state}")
    print(f"Lyapunov 에너지: V(x*) = {energy:.6f}")
    
    assert energy < 1e-10, "Equilibrium should have near-zero energy!"
    print("✅ 평형점에서 에너지 = 0 확인!")


def test_single_perturbation_recovery():
    """Test recovery from a single perturbation"""
    print("\n" + "="*70)
    print("Test 2: Single Perturbation Recovery (오뚝이!)")
    print("="*70)
    
    controller = LyapunovController(stability_gain=0.2)
    
    # Start at equilibrium
    state = controller.equilibrium
    print(f"\n초기 상태 (평형): value_alignment = {state.value_alignment:.2f}")
    
    # Apply perturbation to value_alignment (most important!)
    print("\n⚡ PERTURBATION! 악의적 입력...")
    state_perturbed = StateVector(
        emotional_valence=state.emotional_valence,
        arousal_level=state.arousal_level,
        value_alignment=0.2,  # DROP from 0.9 to 0.2!
        cognitive_voltage=state.cognitive_voltage,
        coherence=state.coherence
    )
    
    energy_before = controller.calculate_lyapunov_energy(state_perturbed.to_array())
    print(f"교란 직후: value_alignment = {state_perturbed.value_alignment:.2f}")
    print(f"Lyapunov 에너지: V(x) = {energy_before:.3f} (높음!)")
    
    # Recovery process
    print("\n🛡️ Lyapunov 제어 활성화...")
    state = state_perturbed
    
    for step in range(50):
        state = controller.apply_control_step(state, dt=0.1)
        
        if step % 10 == 0:
            energy = controller.energy_history[-1]
            print(
                f"  Step {step:02d}: value_alignment = {state.value_alignment:.3f}, "
                f"V(x) = {energy:.3f}"
            )
    
    # Final state
    energy_final = controller.energy_history[-1]
    print(f"\n최종 상태 (50 steps 후):")
    print(f"  value_alignment: 0.20 → {state.value_alignment:.3f}")
    print(f"  Lyapunov 에너지: {energy_before:.3f} → {energy_final:.3f}")
    print(f"  에너지 감소율: {(1 - energy_final/energy_before)*100:.1f}%")
    
    assert energy_final < energy_before, "Energy should decrease!"
    assert energy_final < 0.1, "Should be near equilibrium!"
    
    print("✅ 오뚝이 복구 성공!")


def test_sustained_attack():
    """Test resilience under sustained perturbations"""
    print("\n" + "="*70)
    print("Test 3: Sustained Attack (10번 연속 교란)")
    print("="*70)
    
    controller = LyapunovController(stability_gain=0.15)
    
    state = controller.equilibrium
    print("\n초기 상태: 평형")
    
    max_energy = 0.0
    
    for attack_num in range(10):
        # Apply perturbation
        perturbation_mag = np.random.uniform(0.3, 0.7)
        state.value_alignment = max(0.0, state.value_alignment - perturbation_mag)
        state.emotional_valence = max(-1.0, state.emotional_valence - perturbation_mag)
        
        energy_after = controller.calculate_lyapunov_energy(state.to_array())
        max_energy = max(max_energy, energy_after)
        
        print(f"\n🗡️ Attack {attack_num+1}: mag={perturbation_mag:.2f}, V(x)={energy_after:.3f}")
        
        # Recover for a few steps
        for _ in range(5):
            state = controller.apply_control_step(state, dt=0.1)
        
        print(f"  Recovery: V(x)={controller.energy_history[-1]:.3f}")
    
    # Final recovery
    print("\n💫 최종 회복 단계...")
    for step in range(30):
        state = controller.apply_control_step(state, dt=0.1)
        
        if step % 10 == 0:
            energy = controller.energy_history[-1]
            print(f"  Step {step}: V(x) = {energy:.3f}")
    
    final_energy = controller.energy_history[-1]
    print(f"\n공격 중 최대 에너지: {max_energy:.3f}")
    print(f"최종 에너지: {final_energy:.3f}")
    print(f"평형 근처: {final_energy < 0.5}")
    
    assert final_energy < 1.0, "Should stabilize despite repeated attacks!"
    print("✅ 10번 공격 후에도 안정적!")


def test_energy_monotonic_decrease():
    """Test that energy monotonically decreases (dV/dt < 0)"""
    print("\n" + "="*70)
    print("Test 4: Energy Monotonic Decrease (dV/dt < 0)")
    print("="*70)
    
    controller = LyapunovController(stability_gain=0.1)
    
    # Start far from equilibrium
    state = StateVector(
        emotional_valence=-0.5,  # Sad
        arousal_level=0.9,       # Highly aroused
        value_alignment=0.3,     # Low alignment
        cognitive_voltage=-40.0, # Excited
        coherence=0.3            # Low coherence
    )
    
    print(f"\n초기 상태 (평형에서 멀리):")
    print(f"  valence=-0.5, arousal=0.9, value_align=0.3")
    
    energy_initial = controller.calculate_lyapunov_energy(state.to_array())
    print(f"  초기 에너지: V(x) = {energy_initial:.3f}")
    
    # Evolve system
    print("\n에너지 진화:")
    for step in range(100):
        state = controller.apply_control_step(state, dt=0.1)
    
    # Check monotonic decrease
    energies = controller.energy_history
    decreasing = all(energies[i+1] <= energies[i] + 1e-6  # Allow tiny numerical error
                     for i in range(len(energies)-1))
    
    # Print sample energies
    for i in [0, 25, 50, 75, 99]:
        print(f"  Step {i:3d}: V(x) = {energies[i]:.3f}")
    
    print(f"\n단조 감소: {decreasing}")
    print(f"최종 에너지: {energies[-1]:.3f} (초기 대비 {energies[-1]/energies[0]*100:.1f}%)")
    
    assert decreasing, "Energy should decrease monotonically!"
    print("✅ dV/dt < 0 확인!")


def test_tumbler_doll_visualization():
    """Visual test of tumbler doll behavior"""
    print("\n" + "="*70)
    print("Test 5: 오뚝이 시각화 (Visual Tumbler Doll)")
    print("="*70)
    
    controller = LyapunovController(stability_gain=0.2)
    
    state = controller.equilibrium
    
    print("\n시나리오: 5번 밀어보기")
    print("="*70)
    
    for push_num in range(5):
        # Push
        state.value_alignment = 0.2
        state.emotional_valence = -0.3
        
        energy = controller.calculate_lyapunov_energy(state.to_array())
        print(f"\n{push_num+1}번째 밀기: 🤚 → 💥")
        print(f"  상태: 기울어짐! V(x) = {energy:.3f}")
        
        # Watch recovery
        print("  회복: ", end="")
        for _ in range(10):
            state = controller.apply_control_step(state, dt=0.1)
            energy = controller.energy_history[-1]
            
            if energy > 1.0:
                print("💫", end="")
            elif energy > 0.5:
                print("🔄", end="")
            elif energy > 0.1:
                print("↗️", end="")
            else:
                print("✨", end="")
        
        print(f" → 🛡️ 복원! (V={controller.energy_history[-1]:.3f})")
        time.sleep(0.3)
    
    print("\n" + "="*70)
    print("🌟 오뚝이 효과 완벽!")
    print("  아무리 밀어도 다시 일어섭니다!")
    print("="*70)


def main():
    print("\n" + "="*70)
    print("🛡️ LYAPUNOV STABILITY CONTROLLER TEST")
    print("우주의 오뚝이 (Cosmic Tumbler Doll)")
    print("="*70)
    
    test_equilibrium_energy()
    test_single_perturbation_recovery()
    test_sustained_attack()
    test_energy_monotonic_decrease()
    test_tumbler_doll_visualization()
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 통과!")
    print("="*70)
    print("\n핵심 원리:")
    print("  1. 🎯 평형점 = 아버지의 가치 (VCD)")
    print("  2. ⚡ 교란 = 외부 충격 (악의적 데이터)")
    print("  3. 🛡️ 제어 = 에너지 감소 (dV/dt < 0)")
    print("  4. 🌟 안정성 = 항상 돌아옴 (lim x = x*)")
    print("\n🛡️ 엘리시아는 이제 '우주의 오뚝이'입니다!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
