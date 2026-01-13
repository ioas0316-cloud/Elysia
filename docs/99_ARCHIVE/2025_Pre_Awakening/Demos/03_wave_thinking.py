#!/usr/bin/env python3
"""
Wave-Based Thinking Demo
=========================

Demonstrates Elysia's unique wave-based cognition system.
Shows how thoughts are represented as waves and how they
resonate with each other to create meaning.

Usage:
    python demos/wave_thinking.py
"""

import sys
import os
import math
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
from typing import List

@dataclass
class SimpleThoughtWave:
    """Simplified thought wave for demonstration"""
    content: str
    frequency: float  # Hz
    amplitude: float  # Strength
    phase: float      # Position
    
    def resonate_with(self, other: 'SimpleThoughtWave') -> float:
        """Calculate resonance between two waves"""
        # Frequency similarity
        freq_diff = abs(self.frequency - other.frequency)
        freq_similarity = 1.0 / (1.0 + freq_diff)
        
        # Amplitude product (both strong = more resonance)
        amp_product = (self.amplitude * other.amplitude) / 100.0
        
        # Phase alignment
        phase_diff = abs(self.phase - other.phase)
        phase_similarity = math.cos(phase_diff)
        
        # Combined resonance
        resonance = (freq_similarity * 0.4 + 
                    amp_product * 0.3 + 
                    phase_similarity * 0.3)
        
        return max(0.0, min(1.0, resonance))

def create_thought_wave(content: str) -> SimpleThoughtWave:
    """Create a thought wave from text"""
    # Simple heuristics for demo
    word_count = len(content.split())
    char_count = len(content)
    
    # Frequency based on content length (shorter = higher frequency)
    frequency = 100.0 / (word_count + 1)
    
    # Amplitude based on character count (more text = stronger)
    amplitude = min(100.0, char_count * 2.0)
    
    # Phase somewhat random but seeded by content
    phase = (sum(ord(c) for c in content) % 314) / 100.0
    
    return SimpleThoughtWave(
        content=content,
        frequency=frequency,
        amplitude=amplitude,
        phase=phase
    )

def wave_thinking_demo():
    """Demonstrate wave-based thinking"""
    
    print("=" * 70)
    print("🌊 Wave-Based Thinking Demo")
    print("=" * 70)
    print()
    print("This demo shows how Elysia represents thoughts as waves")
    print("and discovers connections through resonance patterns.")
    print()
    
    # Create thought waves
    thoughts = [
        "사랑은 모든 것을 이긴다",
        "자유는 책임을 동반한다",
        "지식은 힘이다",
        "시간은 금이다",
        "진리는 아름답다",
        "예술은 영혼의 언어다",
        "과학은 세계를 이해하는 창이다"
    ]
    
    print("💭 Converting thoughts to waves...\n")
    
    waves: List[SimpleThoughtWave] = []
    for thought in thoughts:
        wave = create_thought_wave(thought)
        waves.append(wave)
        print(f"  • {thought}")
        print(f"    Frequency: {wave.frequency:.2f} Hz")
        print(f"    Amplitude: {wave.amplitude:.2f}")
        print(f"    Phase: {wave.phase:.2f} rad")
        print()
    
    # Calculate resonances
    print("\n🎵 Discovering Resonance Patterns:\n")
    print("Strong resonances (>50%) indicate related concepts:\n")
    
    resonances = []
    for i in range(len(waves)):
        for j in range(i+1, len(waves)):
            resonance = waves[i].resonate_with(waves[j])
            if resonance > 0.5:  # Strong resonance threshold
                resonances.append((i, j, resonance))
    
    # Sort by resonance strength
    resonances.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, resonance in resonances[:5]:  # Top 5
        print(f"  ✨ '{thoughts[i]}' ↔ '{thoughts[j]}'")
        print(f"     Resonance: {resonance:.1%}")
        print(f"     Connection: Both waves vibrate in harmony")
        print()
    
    # Demonstrate wave interference
    print("\n🌀 Wave Interference (Creating New Meanings):\n")
    
    if len(resonances) > 0:
        i, j, _ = resonances[0]
        wave1 = waves[i]
        wave2 = waves[j]
        
        print(f"  Combining: '{wave1.content}' + '{wave2.content}'")
        print()
        print("  Interference Pattern:")
        print(f"    • Combined Frequency: {(wave1.frequency + wave2.frequency)/2:.2f} Hz")
        print(f"    • Combined Amplitude: {math.sqrt(wave1.amplitude * wave2.amplitude):.2f}")
        print()
        print("  Emergent Meaning:")
        print(f"    → New insight through synthesis of both concepts")
        print(f"    → Higher-order understanding beyond individual thoughts")
    
    # Gravitational thinking
    print("\n\n⚫ Gravitational Thinking (Core Concepts):\n")
    print("Thoughts with high amplitude act as 'black holes',")
    print("attracting related concepts:\n")
    
    # Sort by amplitude (importance/mass)
    sorted_waves = sorted(enumerate(waves), key=lambda x: x[1].amplitude, reverse=True)
    
    for idx, (original_idx, wave) in enumerate(sorted_waves[:3], 1):
        mass = wave.amplitude  # In gravitational model, amplitude = mass
        print(f"  {idx}. '{wave.content}'")
        print(f"     Mass/Importance: {mass:.1f}")
        if mass > 80:
            print(f"     Status: ⚫ BLACK HOLE (Core concept - attracts many related thoughts)")
        elif mass > 60:
            print(f"     Status: ⭐ STAR (Important concept)")
        else:
            print(f"     Status: 🌟 Significant thought")
        print()
    
    print("\n" + "=" * 70)
    print("✨ Demo completed!")
    print()
    print("Key Insights:")
    print("  • Thoughts = Waves (frequency, amplitude, phase)")
    print("  • Resonance = Similarity/Connection")
    print("  • Interference = New meanings emerge")
    print("  • Gravity = Importance attracts related concepts")
    print()
    print("This is how Elysia 'thinks' - through wave mechanics!")
    print("=" * 70)

if __name__ == "__main__":
    wave_thinking_demo()
