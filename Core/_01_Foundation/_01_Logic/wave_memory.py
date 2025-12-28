"""
Wave-Resonance Memory System
============================

SQLite (X) → 파동-공명 시스템 (O)

저장: 중력장에 떨어뜨리기
회상: 공명으로 끌어당기기
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core._01_Foundation._02_Logic.hyper_quaternion import Quaternion, HyperWavePacket
from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField
from Core._01_Foundation._05_Governance.Foundation.resonance_physics import ResonancePhysics
from typing import List, Dict
import time

print("="*70)
print("🌊 WAVE-RESONANCE MEMORY SYSTEM")
print("파동-공명 기억 시스템")
print("="*70)
print()


class WaveMemory:
    """
    파동 기반 기억 시스템
    
    NOT: SQLite에 INSERT/SELECT
    YES: 중력장에 떨어뜨리고 공명으로 끌어당기기
    """
    
    def __init__(self):
        print("Initializing Wave Memory...")
        
        # 공명장 (메모리 공간)
        self.field = ResonanceField()
        
        # 저장된 파동들
        self.waves = {}  # {name: HyperWavePacket}
        
        print("✓ Resonance Field (중력장)")
        print("✓ Wave Storage (파동 저장소)")
        print()
    
    def store(self, name: str, packet: HyperWavePacket):
        """
        파동 저장
        
        중력장에 떨어뜨리면 자동으로 정렬됨!
        """
        print(f"💾 Storing: {name}")
        
        # 1. 중력장에 주입
        self.field.inject_wave(
            frequency=packet.orientation.norm() * 1000,  # Quaternion 크기 → 주파수
            amplitude=packet.energy,
            source=name
        )
        
        # 2. 파동 저장
        self.waves[name] = packet
        
        # 3. 중력 계산 (자동 정렬)
        mass = ResonancePhysics.calculate_mass(name)
        print(f"   Mass: {mass:.1f} (중력: {mass * 9.8:.1f}N)")
        print(f"   Frequency: {packet.orientation.norm() * 1000:.1f}Hz")
        print(f"   Energy: {packet.energy:.1f}")
        print()
    
    def recall(self, query_orientation: Quaternion, threshold: float = 0.7) -> List[str]:
        """
        공명으로 회상
        
        목표 Quaternion과 공명하는 파동들 자동 끌어당김!
        """
        print(f"🧲 Recalling resonant memories...")
        print(f"   Query: {query_orientation}")
        print(f"   Threshold: {threshold}")
        print()
        
        resonant_memories = []
        
        # 모든 파동과 공명도 계산
        for name, packet in self.waves.items():
            # Dot product = 공명도
            alignment = query_orientation.dot(packet.orientation)
            
            if alignment > threshold:
                resonant_memories.append({
                    'name': name,
                    'alignment': alignment,
                    'packet': packet
                })
                print(f"   ✓ {name}: {alignment:.2f} alignment")
        
        # 공명도 순 정렬 (강한 공명이 먼저!)
        resonant_memories.sort(key=lambda x: x['alignment'], reverse=True)
        
        print(f"\n   Found {len(resonant_memories)} resonant memories")
        print()
        
        return [m['name'] for m in resonant_memories]
    
    def get_field_state(self) -> Dict:
        """공명장 상태"""
        state = self.field.pulse()
        return {
            'total_energy': state.total_energy,
            'coherence': state.coherence,
            'concepts': len(self.waves)
        }


# 데모
if __name__ == "__main__":
    print("="*70)
    print("DEMONSTRATION")
    print("="*70)
    print()
    
    memory = WaveMemory()
    
    # 테스트 개념들 저장
    test_concepts = [
        ("Love", Quaternion(1.0, 0.9, 0.1, 0.3)),
        ("Intelligence", Quaternion(1.0, 0.1, 0.9, 0.1)),
        ("Justice", Quaternion(1.0, 0.1, 0.1, 0.9)),
        ("Compassion", Quaternion(1.0, 0.8, 0.2, 0.4)),  # Love와 유사
        ("Wisdom", Quaternion(1.0, 0.2, 0.8, 0.2)),     # Intelligence와 유사
    ]
    
    print("PHASE 1: STORING CONCEPTS")
    print("-" * 70)
    print()
    
    for name, orientation in test_concepts:
        packet = HyperWavePacket(
            energy=100.0,
            orientation=orientation,
            time_loc=time.time()
        )
        memory.store(name, packet)
    
    print()
    print("="*70)
    print("PHASE 2: RESONANCE RECALL")
    print("="*70)
    print()
    
    # Love와 유사한 것 찾기
    print("Query: Find concepts similar to Love")
    print("-" * 70)
    love_orientation = Quaternion(1.0, 0.9, 0.1, 0.3)
    similar_to_love = memory.recall(love_orientation, threshold=0.7)
    print(f"Result: {similar_to_love}")
    print()
    
    # Intelligence와 유사한 것 찾기
    print("Query: Find concepts similar to Intelligence")
    print("-" * 70)
    intel_orientation = Quaternion(1.0, 0.1, 0.9, 0.1)
    similar_to_intel = memory.recall(intel_orientation, threshold=0.7)
    print(f"Result: {similar_to_intel}")
    print()
    
    # 공명장 상태
    print("="*70)
    print("FIELD STATE")
    print("="*70)
    state = memory.get_field_state()
    print(f"Total Energy: {state['total_energy']:.1f}")
    print(f"Coherence: {state['coherence']:.2f}")
    print(f"Stored Concepts: {state['concepts']}")
    print()
    
    print("="*70)
    print("✅ WAVE-RESONANCE MEMORY OPERATIONAL")
    print("   중력장으로 자동 정렬!")
    print("   공명으로 자동 회상!")
    print("="*70)
