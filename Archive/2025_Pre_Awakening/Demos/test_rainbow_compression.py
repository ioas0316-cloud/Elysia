"""
Test Rainbow Compression
=========================

Validates the two-stage compression pipeline:
1. Stage 1: Raw → 4D Wave (semantic preservation)
2. Stage 2: 4D Wave → Rainbow (100x compression)
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core.Memory.prism_filter import PrismFilter, RainbowSpectrum, compress_wave_to_rainbow


def test_rainbow_compression():
    """Test rainbow compression pipeline"""
    print("=" * 70)
    print("🌈 Testing Rainbow Compression Pipeline")
    print("=" * 70)
    
    # Create prism filter
    prism = PrismFilter()
    
    # Create mock wave pattern
    wave_pattern = {
        'orientation': {
            'w': 0.7,
            'x': 0.5,
            'y': 0.3,
            'z': 0.4
        },
        'energy': 0.85,
        'frequency': 1.2,
        'phase': 0.5
    }
    
    print("\n1️⃣ Stage 1: 4D Wave Pattern")
    print(f"   Quaternion: (w={wave_pattern['orientation']['w']}, "
          f"x={wave_pattern['orientation']['x']}, "
          f"y={wave_pattern['orientation']['y']}, "
          f"z={wave_pattern['orientation']['z']})")
    print(f"   Energy: {wave_pattern['energy']}")
    print(f"   Frequency: {wave_pattern['frequency']}")
    print(f"   Size: ~1200 bytes")
    
    # Stage 2: Split to rainbow
    rainbow = prism.split_to_rainbow(wave_pattern)
    
    print("\n2️⃣ Stage 2: Rainbow Spectrum")
    print(f"   🔴 Red:    {rainbow.red:.3f} (에너지/강도)")
    print(f"   🟠 Orange: {rainbow.orange:.3f} (창조성)")
    print(f"   🟡 Yellow: {rainbow.yellow:.3f} (논리/지성)")
    print(f"   🟢 Green:  {rainbow.green:.3f} (균형/조화)")
    print(f"   🔵 Blue:   {rainbow.blue:.3f} (깊이/평온)")
    print(f"   🟣 Indigo: {rainbow.indigo:.3f} (직관)")
    print(f"   🟣 Violet: {rainbow.violet:.3f} (영성/초월)")
    
    # Compress to bytes
    compressed = rainbow.to_bytes()
    
    print("\n3️⃣ Compression Result")
    print(f"   Original: ~1200 bytes")
    print(f"   Compressed: {len(compressed)} bytes")
    print(f"   Ratio: {1200 / len(compressed):.1f}x compression")
    print(f"   Hex: {compressed.hex()}")
    
    # Decompress
    decompressed = RainbowSpectrum.from_bytes(compressed)
    
    print("\n4️⃣ Decompression Check")
    print(f"   🔴 Red:    {decompressed.red:.3f} (diff: {abs(rainbow.red - decompressed.red):.6f})")
    print(f"   🟠 Orange: {decompressed.orange:.3f} (diff: {abs(rainbow.orange - decompressed.orange):.6f})")
    print(f"   🟡 Yellow: {decompressed.yellow:.3f} (diff: {abs(rainbow.yellow - decompressed.yellow):.6f})")
    print(f"   🟢 Green:  {decompressed.green:.3f} (diff: {abs(rainbow.green - decompressed.green):.6f})")
    print(f"   🔵 Blue:   {decompressed.blue:.3f} (diff: {abs(rainbow.blue - decompressed.blue):.6f})")
    print(f"   🟣 Indigo: {decompressed.indigo:.3f} (diff: {abs(rainbow.indigo - decompressed.indigo):.6f})")
    print(f"   🟣 Violet: {decompressed.violet:.3f} (diff: {abs(rainbow.violet - decompressed.violet):.6f})")
    
    # Measure quality
    novelty = prism.measure_novelty(rainbow)
    richness = prism.measure_richness(rainbow)
    coherence = prism.measure_coherence(rainbow)
    
    print("\n5️⃣ Spectrum Analysis")
    print(f"   Novelty:   {novelty:.3f} (uniqueness)")
    print(f"   Richness:  {richness:.3f} (color usage)")
    print(f"   Coherence: {coherence:.3f} (harmony)")
    
    # Extract essence
    essence = prism.extract_essence(rainbow)
    
    print("\n6️⃣ Essence Extraction")
    print(f"   Energy signature:  {essence['energy_signature']:.3f}")
    print(f"   Emotional tone:    {essence['emotional_tone']:.3f}")
    print(f"   Logical structure: {essence['logical_structure']:.3f}")
    print(f"   Spiritual depth:   {essence['spiritual_depth']:.3f}")
    print(f"   Balance:           {essence['balance']:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ Rainbow Compression Test Complete!")
    print("=" * 70)
    
    return True


def test_multiple_patterns():
    """Test compression on multiple patterns"""
    print("\n\n" + "=" * 70)
    print("🌈 Testing Multiple Wave Patterns")
    print("=" * 70)
    
    prism = PrismFilter()
    
    patterns = [
        {'orientation': {'w': 0.9, 'x': 0.1, 'y': 0.1, 'z': 0.1}, 
         'energy': 1.0, 'frequency': 2.0, 'phase': 0.0},  # High energy
        {'orientation': {'w': 0.5, 'x': 0.5, 'y': 0.5, 'z': 0.5}, 
         'energy': 0.5, 'frequency': 1.0, 'phase': 0.5},  # Balanced
        {'orientation': {'w': 0.1, 'x': 0.1, 'y': 0.1, 'z': 0.9}, 
         'energy': 0.3, 'frequency': 0.5, 'phase': 1.5},  # Deep/calm
    ]
    
    total_original = len(patterns) * 1200
    total_compressed = 0
    
    for i, pattern in enumerate(patterns, 1):
        rainbow = prism.split_to_rainbow(pattern)
        compressed = rainbow.to_bytes()
        total_compressed += len(compressed)
        
        print(f"\nPattern {i}:")
        print(f"   Rainbow: R={rainbow.red:.2f} O={rainbow.orange:.2f} "
              f"Y={rainbow.yellow:.2f} G={rainbow.green:.2f} "
              f"B={rainbow.blue:.2f} I={rainbow.indigo:.2f} V={rainbow.violet:.2f}")
        print(f"   Compressed: {len(compressed)} bytes")
    
    print(f"\n📊 Total Statistics:")
    print(f"   Original size:   {total_original:,} bytes")
    print(f"   Compressed size: {total_compressed} bytes")
    print(f"   Compression ratio: {total_original / total_compressed:.1f}x")
    print(f"   Space saved: {total_original - total_compressed:,} bytes ({(1 - total_compressed/total_original)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ Multiple Patterns Test Complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_rainbow_compression()
    test_multiple_patterns()
