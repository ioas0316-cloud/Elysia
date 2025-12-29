"""
Test Neuronal Thought Dynamics

Demonstrates voltage accumulation and firing behavior.
"""

import time
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.neuron_cortex import CognitiveNeuron, ThoughtAccumulator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_single_neuron():
    """Test individual neuron firing"""
    print("\n" + "="*70)
    print("Test 1: Single Neuron - Voltage Accumulation & Firing")
    print("="*70)
    
    neuron = CognitiveNeuron(neuron_id="test_neuron", threshold=-55.0)
    
    print(f"\n초기 상태: V={neuron.voltage:.2f}mV (휴지전위)")
    
    # Accumulate stimuli
    print("\n자극 투입 중...")
    for i in range(5):
        neuron.accumulate_stimulus(
            strength=5.0,
            value_alignment=0.9,  # 높은 가치 정렬 (Na 채널 개방)
            duration=0.2
        )
        
        print(f"  자극 {i+1}: V={neuron.voltage:.2f}mV", end="")
        
        if neuron.check_firing():
            print(" → 🔥 발화!")
            break
        else:
            print(f" (임계값 {neuron.V_threshold}mV까지 {neuron.V_threshold - neuron.voltage:.2f}mV 남음)")
    
    state = neuron.get_state()
    print(f"\n최종 상태:")
    print(f"  전압: {state['voltage']:.2f}mV")
    print(f"  발화율: {state['firing_rate']:.2f} Hz")
    print(f"  Na 활성화 (accept): {state['m']:.3f}")
    print(f"  K 활성화 (reject): {state['n']:.3f}")


def test_value_rejection():
    """Test value-opposed stimulus (K channel opens)"""
    print("\n" + "="*70)
    print("Test 2: Value Rejection - K Channel Opening")
    print("="*70)
    
    neuron = CognitiveNeuron(neuron_id="reject_test")
    
    print(f"\n초기 전압: {neuron.voltage:.2f}mV")
    
    # Apply value-opposed stimulus
    print("\n가치 불일치 자극 투입...")
    for i in range(3):
        neuron.accumulate_stimulus(
            strength=5.0,
            value_alignment=0.1,  # 낮은 가치 정렬 (K 채널 개방)
            duration=0.2
        )
        
        print(f"  자극 {i+1}: V={neuron.voltage:.2f}mV (K 채널이 전압을 낮춤)")
    
    print(f"\n결과: 전압이 {neuron.voltage:.2f}mV로 감소 (발화하지 않음)")


def test_refractory_period():
    """Test refractory period after firing"""
    print("\n" + "="*70)
    print("Test 3: Refractory Period - No Firing During Rest")
    print("="*70)
    
    neuron = CognitiveNeuron(neuron_id="refract_test", refractory_period=2.0)
    
    # Fire once
    print("\n첫 번째 발화 유도...")
    for _ in range(5):
        neuron.accumulate_stimulus(strength=5.0, value_alignment=0.9)
        if neuron.check_firing():
            print("✅ 발화 성공!")
            break
    
    # Try to fire again immediately
    print("\n즉시 재발화 시도...")
    for i in range(3):
        neuron.accumulate_stimulus(strength=10.0, value_alignment=1.0)  # 강한 자극
        result = neuron.check_firing()
        
        if neuron.in_refractory:
            print(f"  시도 {i+1}: ⛔ 불응기 - 발화 불가")
        else:
            print(f"  시도 {i+1}: ✅ 불응기 종료 - 발화 가능")
            break
        
        time.sleep(0.5)


def test_thought_accumulator():
    """Test multi-neuron thought accumulation"""
    print("\n" + "="*70)
    print("Test 4: Thought Accumulator - '음... 아!' Process")
    print("="*70)
    
    accumulator = ThoughtAccumulator(num_neurons=5)
    
    print("\n사용자: '사랑이 뭐야?'")
    print("엘리시아: 음... (생각 중)")
    
    thought_complete = False
    thinking_steps = 0
    
    # Simulate thinking process
    for step in range(10):
        thinking_steps += 1
        
        result = accumulator.process_stimulus(
            content="사랑은 서로를 있는 그대로 받아들이는 것입니다",
            strength=3.0,
            value_alignment=0.85,
        )
        
        if result:
            print(f"\n⚡ 아! (생각 완료)")
            print(f"엘리시아: {result}")
            thought_complete = True
            break
        else:
            print(".", end="", flush=True)
            time.sleep(0.2)
    
    if not thought_complete:
        print(f"\n⏱️ (생각 시간 초과)")
        print(f"엘리시아: {accumulator.force_output()}")
    
    stats = accumulator.get_statistics()
    print(f"\n통계:")
    print(f"  평균 전압: {stats['avg_voltage']:.2f}mV")
    print(f"  통합 뉴런 전압: {stats['integration_voltage']:.2f}mV")
    print(f"  평균 발화율: {stats['avg_firing_rate']:.2f} Hz")


def main():
    print("\n" + "="*70)
    print("🧠 호지킨-헉슬리 인지 모델 테스트")
    print("="*70)
    
    test_single_neuron()
    test_value_rejection()
    test_refractory_period()
    test_thought_accumulator()
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 완료!")
    print("="*70)
    print("\n핵심 원리:")
    print("  1. 💭 자극이 쌓여서 전압 상승 (voltage accumulation)")
    print("  2. ⚡ 임계값 도달 시 발화 (threshold firing)")
    print("  3. 😌 발화 후 휴식 필요 (refractory period)")
    print("  4. 🚪 가치 정렬이 Na/K 채널 제어 (value gating)")
    print("\n🌟 엘리시아는 이제 '생각하는' 존재입니다!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
