import os
import sys
import time
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ingestion.natural_mapper import NaturalMapper
from synaptic_architecture.organism import DirectMappingOrganism

def calculate_chaos_score(tensions):
    """
    Calculates a Chaos/Variance score from NaturalMapper tensions.
    Math_scalar represents perfect alignment (0 variance).
    Space, Lang, Time represent deviations/friction.
    """
    total_tension = sum(tensions.values())
    if total_tension == 0:
        return 0.0
    
    # Mathematical scalar means 0 bits XOR difference -> Perfect harmony
    harmony = tensions.get("math_scalar", 0)
    
    # Chaos is the proportion of non-harmonic tensions
    chaos = (total_tension - harmony) / total_tension
    return chaos

def bytes_to_uint64(byte_chunk):
    """
    Safely converts up to 8 bytes into a 64-bit unsigned integer.
    """
    padded = bytearray(8)
    for i in range(min(len(byte_chunk), 8)):
        padded[i] = byte_chunk[i]
    return np.frombuffer(padded, dtype=np.uint64)[0]

def run_unified_resonance():
    print("==================================================================")
    print(" [Elysia] Unified Thermal Resonance Observation")
    print(" (Natural Mapper Tension -> Thermal Scheduler -> Synaptic Gene)")
    print("==================================================================\n")

    mapper = NaturalMapper(terrain_size=256)
    organism = DirectMappingOrganism(resolution=256)

    # Initialize terrain
    mapper.set_terrain(b"Elysia_Origin_Seed")

    # Let's observe a pattern. 
    # Phase 1: High noise (chaos) -> Will cause high temperature.
    print("\n--- [Phase 1] High Noise Stream (Induces Chaos & High Temperature) ---")
    np.random.seed(42)
    noise_stream = np.random.bytes(128)
    
    chunk_size = 8 # 64-bit mapping
    
    for i in range(0, len(noise_stream), chunk_size):
        chunk = noise_stream[i:i+chunk_size]
        if len(chunk) < chunk_size: continue
        
        # 1. Natural Mapper Observation
        tensions = mapper.map_and_observe(chunk)
        chaos_score = calculate_chaos_score(tensions)
        
        # 2. Thermal Feedback (T = 0.1 + Chaos * 10.0)
        # High chaos -> High Temp (> 2.0 triggers jitter mask)
        T = 0.1 + (chaos_score * 10.0)
        organism.scheduler.set_temperature(T)
        
        print(f"\n[Observe] Chaos: {chaos_score:.2f} -> Setting Temperature: {T:.2f}")
        
        # 3. Flow into Synaptic Architecture
        wave = bytes_to_uint64(chunk)
        organism.flow(wave)
        time.sleep(0.05)


    # Phase 2: Ordered, repeating pattern -> Will cause resonance & cooling
    print("\n--- [Phase 2] Harmonious Repeating Stream (Induces Resonance & Cooling) ---")
    # Repeated "Elysia" bytes
    harmony_stream = (b"Elysia  " * 16)
    
    # We want to show that over time, the Natural Mapper adapts (because terrain updates)
    # and chaos score drops to 0, causing T to drop to 0.1.
    for i in range(0, len(harmony_stream), chunk_size):
        chunk = harmony_stream[i:i+chunk_size]
        if len(chunk) < chunk_size: continue
        
        # 1. Natural Mapper Observation
        tensions = mapper.map_and_observe(chunk)
        chaos_score = calculate_chaos_score(tensions)
        
        # 2. Thermal Feedback
        T = 0.1 + (chaos_score * 10.0)
        organism.scheduler.set_temperature(T)
        
        print(f"\n[Observe] Chaos: {chaos_score:.2f} -> Setting Temperature: {T:.2f}")
        
        # 3. Flow into Synaptic Architecture
        wave = bytes_to_uint64(chunk)
        organism.flow(wave)
        time.sleep(0.05)

    print("\n==================================================================")
    print(" [Observation Complete] Unified Loop fully synchronized.")
    print("==================================================================")

if __name__ == "__main__":
    run_unified_resonance()
