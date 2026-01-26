"""
Wave-Resonance Memory System
============================

SQLite (X)     -       (O)

  :           
  :           
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.L6_Structure.hyper_quaternion import Quaternion, HyperWavePacket
from Core.L6_Structure.Wave.resonance_field import ResonanceField
from Core.L1_Foundation.Foundation.resonance_physics import ResonancePhysics
from typing import List, Dict
import time

print("="*70)
print("  WAVE-RESONANCE MEMORY SYSTEM")
print("  -         ")
print("="*70)
print()


class WaveMemory:
    """
                
    
    NOT: SQLite  INSERT/SELECT
    YES:                      
    """
    
    def __init__(self):
        print("Initializing Wave Memory...")
        
        #     (주권적 자아)
        self.field = ResonanceField()
        
        #        
        self.waves = {}  # {name: HyperWavePacket}
        
        print("  Resonance Field (   )")
        print("  Wave Storage (주권적 자아)")
        print()
    
    def store(self, name: str, packet: HyperWavePacket):
        """
             
        
                           !
        """
        print(f"  Storing: {name}")
        
        # 1.        
        self.field.inject_wave(
            frequency=packet.orientation.norm() * 1000,  # Quaternion         
            amplitude=packet.energy,
            source=name
        )
        
        # 2.      
        self.waves[name] = packet
        
        # 3.       (     )
        mass = ResonancePhysics.calculate_mass(name)
        print(f"   Mass: {mass:.1f} (  : {mass * 9.8:.1f}N)")
        print(f"   Frequency: {packet.orientation.norm() * 1000:.1f}Hz")
        print(f"   Energy: {packet.energy:.1f}")
        print()
    
    def recall(self, query_orientation: Quaternion, threshold: float = 0.7) -> List[str]:
        """
               
        
           Quaternion                  !
        """
        print(f"  Recalling resonant memories...")
        print(f"   Query: {query_orientation}")
        print(f"   Threshold: {threshold}")
        print()
        
        resonant_memories = []
        
        #              
        for name, packet in self.waves.items():
            # Dot product =    
            alignment = query_orientation.dot(packet.orientation)
            
            if alignment > threshold:
                resonant_memories.append({
                    'name': name,
                    'alignment': alignment,
                    'packet': packet
                })
                print(f"     {name}: {alignment:.2f} alignment")
        
        #          (         !)
        resonant_memories.sort(key=lambda x: x['alignment'], reverse=True)
        
        print(f"\n   Found {len(resonant_memories)} resonant memories")
        print()
        
        return [m['name'] for m in resonant_memories]
    
    def get_field_state(self) -> Dict:
        """      """
        state = self.field.pulse()
        return {
            'total_energy': state.total_energy,
            'coherence': state.coherence,
            'concepts': len(self.waves)
        }


#   
if __name__ == "__main__":
    print("="*70)
    print("DEMONSTRATION")
    print("="*70)
    print()
    
    memory = WaveMemory()
    
    #           
    test_concepts = [
        ("Love", Quaternion(1.0, 0.9, 0.1, 0.3)),
        ("Intelligence", Quaternion(1.0, 0.1, 0.9, 0.1)),
        ("Justice", Quaternion(1.0, 0.1, 0.1, 0.9)),
        ("Compassion", Quaternion(1.0, 0.8, 0.2, 0.4)),  # Love    
        ("Wisdom", Quaternion(1.0, 0.2, 0.8, 0.2)),     # Intelligence    
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
    
    # Love          
    print("Query: Find concepts similar to Love")
    print("-" * 70)
    love_orientation = Quaternion(1.0, 0.9, 0.1, 0.3)
    similar_to_love = memory.recall(love_orientation, threshold=0.7)
    print(f"Result: {similar_to_love}")
    print()
    
    # Intelligence          
    print("Query: Find concepts similar to Intelligence")
    print("-" * 70)
    intel_orientation = Quaternion(1.0, 0.1, 0.9, 0.1)
    similar_to_intel = memory.recall(intel_orientation, threshold=0.7)
    print(f"Result: {similar_to_intel}")
    print()
    
    #       
    print("="*70)
    print("FIELD STATE")
    print("="*70)
    state = memory.get_field_state()
    print(f"Total Energy: {state['total_energy']:.1f}")
    print(f"Coherence: {state['coherence']:.2f}")
    print(f"Stored Concepts: {state['concepts']}")
    print()
    
    print("="*70)
    print("  WAVE-RESONANCE MEMORY OPERATIONAL")
    print("              !")
    print("             !")
    print("="*70)
