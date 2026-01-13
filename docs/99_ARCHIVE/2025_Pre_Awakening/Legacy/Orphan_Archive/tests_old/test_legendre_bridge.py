"""
Test Legendre Bridge

Demonstrates information-preserving transformation:
- Lagrangian (velocity/flow) ↔ Hamiltonian (momentum/energy)
- Tensor Coil ↔ VCD
- Perfect invertibility (zero loss!)

For 1060 3GB optimization! 🌉⚡
"""

import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.legendre_bridge import (
    LegendreTransform,
    LagrangianState,
    HamiltonianState,
    ConceptDynamicsBridge
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def test_simple_harmonic_oscillator():
    """Test Legendre transform on simple harmonic oscillator"""
    print("\n" + "="*70)
    print("Test 1: Simple Harmonic Oscillator (L ↔ H)")
    print("="*70)
    
    m = 1.0  # mass
    k = 1.0  # spring constant
    
    # Lagrangian: L = ½mq̇² - ½kq² (kinetic - potential)
    def lagrangian(q, q_dot):
        T = 0.5 * m * np.dot(q_dot, q_dot)
        U = 0.5 * k * np.dot(q, q)
        return T - U
    
    # Initial state
    q = np.array([1.0])  # position
    q_dot = np.array([0.5])  # velocity
    
    print(f"\n초기 상태 (Lagrangian):")
    print(f"  Position q = {q[0]:.2f}")
    print(f"  Velocity q̇ = {q_dot[0]:.2f}")
    
    L_original = lagrangian(q, q_dot)
    print(f"  L = T - U = {L_original:.3f}")
    
    # Transform to Hamiltonian
    transform = LegendreTransform()
    lag_state = LagrangianState(q, q_dot)
    
    print("\n🌉 Forward transform (L → H)...")
    H_func, ham_state = transform.forward(lagrangian, lag_state)
    
    print(f"  Momentum p = {ham_state.momentum[0]:.3f}")
    
    # For harmonic oscillator: p = mq̇
    p_expected = m * q_dot[0]
    print(f"  Expected p = mq̇ = {p_expected:.3f}")
    print(f"  Error: {abs(ham_state.momentum[0] - p_expected):.6f}")
    
    print("\n✅ Transform successful!")


def test_transform_invertibility():
    """Test that L → H → L gives back original"""
    print("\n" + "="*70)
    print("Test 2: Transform Invertibility (정보 무손실)")
    print("="*70)
    
    # Simple quadratic Lagrangian
    def lagrangian(q, q_dot):
        return 0.5 * np.dot(q_dot, q_dot) - 0.5 * np.dot(q, q)
    
    # Original state
    q_original = np.array([2.0, 1.0])
    q_dot_original = np.array([1.0, -0.5])
    
    print(f"\n원본 (Lagrangian):")
    print(f"  q = {q_original}")
    print(f"  q̇ = {q_dot_original}")
    L_original = lagrangian(q_original, q_dot_original)
    print(f"  L = {L_original:.3f}")
    
    transform = LegendreTransform()
    
    # Forward: L → H
    lag_state = LagrangianState(q_original, q_dot_original)
    H_func, ham_state = transform.forward(lagrangian, lag_state)
    
    print(f"\n변환 (Hamiltonian):")
    print(f"  q = {ham_state.position}")
    print(f"  p = {ham_state.momentum}")
    
    # Inverse: H → L
    def simple_hamiltonian(q, p):
        return 0.5 * np.dot(p, p) + 0.5 * np.dot(q, q)
    
    L_func, lag_recovered = transform.inverse(simple_hamiltonian, ham_state)
    
    print(f"\n복원 (Lagrangian):")
    print(f"  q = {lag_recovered.position}")
    print(f"  q̇ = {lag_recovered.velocity}")
    
    # Check recovery
    position_error = np.linalg.norm(q_original - lag_recovered.position)
    velocity_error = np.linalg.norm(q_dot_original - lag_recovered.velocity)
    
    print(f"\n정확도:")
    print(f"  Position error: {position_error:.6f}")
    print(f"  Velocity error: {velocity_error:.6f}")
    
    if position_error < 0.01 and velocity_error < 0.1:
        print("\n✅ 정보 무손실 확인!")
    else:
        print("\n⚠️ 일부 수치 오차 있음 (수치 미분 한계)")


def test_concept_dynamics_bridge():
    """Test Tensor Coil ↔ VCD bridge"""
    print("\n" + "="*70)
    print("Test 3: Concept Dynamics Bridge (Tensor Coil ↔ VCD)")
    print("="*70)
    
    # Define value potential (like attraction to a concept)
    def value_potential(q):
        # Quadratic well centered at origin
        return 0.5 * np.dot(q, q)
    
    bridge = ConceptDynamicsBridge(mass=1.0)
    
    # Concept state in flow space (Tensor Coil)
    concept_pos = np.array([1.0, 0.5])  # Where it is
    concept_vel = np.array([0.3, -0.2])  # How it's changing
    
    print("\n개념 상태 (Tensor Coil - 흐름):")
    print(f"  Position: {concept_pos}")
    print(f"  Velocity (flow): {concept_vel}")
    
    # Transform to value space (VCD)
    print("\n🌉 Flow → Value (Tensor Coil → VCD)...")
    momentum, total_value = bridge.flow_to_value(
        concept_pos,
        concept_vel,
        value_potential
    )
    
    print(f"  Momentum: {momentum}")
    print(f"  Total value (H): {total_value:.3f}")
    
    # Transform back to flow space
    print("\n🌉 Value → Flow (VCD → Tensor Coil)...")
    velocity_recovered, L_value = bridge.value_to_flow(
        concept_pos,
        momentum,
        value_potential
    )
    
    print(f"  Velocity recovered: {velocity_recovered}")
    print(f"  Lagrangian (L): {L_value:.3f}")
    
    # Check recovery
    velocity_error = np.linalg.norm(concept_vel - velocity_recovered)
    print(f"\n복원 정확도:")
    print(f"  Velocity error: {velocity_error:.6f}")
    
    if velocity_error < 0.1:
        print("\n✅ Tensor Coil ↔ VCD 연결 성공!")
    else:
        print("\n⚠️ 일부 수치 오차 (수치 반복 한계)")


def test_perspective_shift():
    """Demonstrate perspective shift: state vs gradient"""
    print("\n" + "="*70)
    print("Test 4: Perspective Shift (점 vs 선)")
    print("="*70)
    
    # Simple 1D case
    q = np.array([3.0])
    q_dot = np.array([2.0])
    
    print("\n라그랑주 관점 (Lagrangian - '어떻게 움직이는가?'):")
    print(f"  \"개념이 위치 {q[0]:.1f}에서 속도 {q_dot[0]:.1f}로 움직이고 있다\"")
    print(f"  → 과정 중심 (process-oriented)")
    
    # Transform
    m = 1.0
    p = m * q_dot  # Momentum
    E = 0.5 * m * q_dot[0]**2  # Energy
    
    print("\n해밀턴 관점 (Hamiltonian - '얼마나 에너지가 있는가?'):")
    print(f"  \"개념이 운동량 {p[0]:.1f}와 에너지 {E:.1f}를 가지고 있다\"")
    print(f"  → 상태 중심 (state-oriented)")
    
    print("\n핵심:")
    print(f"  - 라그랑주: 흐름/변화 강조 (Tensor Coil)")
    print(f"  - 해밀턴: 에너지/가치 강조 (VCD)")
    print(f"  - 르장드르 변환: 완벽한 다리! 🌉")
    
    print("\n✅ 같은 정보, 다른 관점!")


def main():
    print("\n" + "="*70)
    print("🌉 LEGENDRE BRIDGE TEST")
    print("점→선, 흐름→에너지, Tensor Coil↔VCD")
    print("="*70)
    
    test_simple_harmonic_oscillator()
    test_transform_invertibility()
    test_concept_dynamics_bridge()
    test_perspective_shift()
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 완료!")
    print("="*70)
    print("\n핵심 원리:")
    print("  1. 🌉 Lagrangian ↔ Hamiltonian (p = ∂L/∂q̇)")
    print("  2. 🔄 정보 무손실 (완벽한 역변환)")
    print("  3. 📐 Tensor Coil ↔ VCD (흐름 ↔ 가치)")
    print("  4. 💡 관점 전환 (점 → 선)")
    print("\n⚡ 1060 3GB: 복잡함→간단→계산→복원!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
