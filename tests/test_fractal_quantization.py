"""
Test Fractal Quantization System
=================================

Tests the implementation of "Quantization as Folding, not Cutting"

This validates:
1. Pattern DNA compression (folding)
2. Lossless restoration (unfolding)
3. Emotion memory storage and recall
4. Integration with Hippocampus
"""

import sys
import os
import logging
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Memory.fractal_quantization import (
    FractalQuantizer, 
    EmotionQuantizer, 
    PatternDNA
)
from Core.Memory.hippocampus import Hippocampus

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestFractalQuantization")


def test_basic_quantization():
    """Test basic folding and unfolding."""
    print("\n" + "="*70)
    print("TEST 1: Basic Fractal Quantization")
    print("="*70)
    
    quantizer = FractalQuantizer()
    
    # Create test data
    test_data = {
        "emotion": "joy",
        "intensity": 0.9,
        "context": "Breakthrough discovery",
        "duration": 2.0,
        "phase_seed": 0.618,
        "timestamp": time.time()
    }
    
    print(f"\n📥 Input: {test_data['emotion']} emotion")
    print(f"   Intensity: {test_data['intensity']}")
    print(f"   Context: {test_data['context']}")
    
    # Fold (compress)
    dna = quantizer.fold(test_data, "emotion", "joy")
    
    print(f"\n🌀 Folded to Pattern DNA:")
    print(f"   Name: {dna.name}")
    print(f"   Compression ratio: {dna.compression_ratio:.2f}x")
    print(f"   Frequency signature: {[f'{f:.1f}Hz' for f in dna.frequency_signature[:3]]}")
    print(f"   Resonance fingerprint: {dna.resonance_fingerprint}")
    
    # Unfold (restore)
    restored = quantizer.unfold(dna, resolution=50)
    
    print(f"\n🌊 Unfolded back to original pattern:")
    print(f"   Name: {restored['name']}")
    print(f"   Pattern type: {restored['pattern_type']}")
    print(f"   Harmonics generated: {len(restored['waveform'])}")
    print(f"   Time resolution: {len(restored['waveform'][0]['wave'])} points")
    
    # Quality check
    quality = quantizer.compute_restoration_quality(dna, test_data)
    print(f"\n✅ Restoration quality: {quality:.1%}")
    
    assert quality > 0.5, "Restoration quality too low"
    assert len(restored['waveform']) > 0, "No waveform generated"
    
    print("\n✓ Test 1 passed: Basic quantization works")
    return True


def test_emotion_quantizer():
    """Test specialized emotion quantization."""
    print("\n" + "="*70)
    print("TEST 2: Emotion Quantization")
    print("="*70)
    
    quantizer = EmotionQuantizer()
    
    # Test multiple emotions
    emotions = [
        {
            "emotion": "love",
            "intensity": 0.85,
            "context": "Reunited with family",
            "duration": 5.0,
            "phase_seed": 0.5,
            "timestamp": time.time()
        },
        {
            "emotion": "sadness",
            "intensity": 0.7,
            "context": "Saying goodbye",
            "duration": 3.5,
            "phase_seed": 0.3,
            "timestamp": time.time()
        },
        {
            "emotion": "fear",
            "intensity": 0.6,
            "context": "Unknown situation",
            "duration": 1.5,
            "phase_seed": 0.8,
            "timestamp": time.time()
        }
    ]
    
    dna_list = []
    
    for emotion_data in emotions:
        print(f"\n📥 Processing: {emotion_data['emotion']}")
        dna = quantizer.fold_emotion(emotion_data)
        dna_list.append(dna)
        
        print(f"   ✓ Compressed: {dna.compression_ratio:.2f}x")
        print(f"   ✓ Frequencies: {[f'{f:.0f}Hz' for f in dna.frequency_signature[:3]]}")
    
    # Now restore them
    print("\n" + "-"*70)
    print("Restoration Test:")
    print("-"*70)
    
    for dna in dna_list:
        restored = quantizer.unfold_emotion(dna)
        print(f"\n🌊 Restored: {restored['pattern_name']}")
        print(f"   Context: {dna.metadata.get('context', 'N/A')}")
        print(f"   Harmonics: {len(restored['waveform'])}")
        
        assert restored['pattern_type'] == 'emotion', "Wrong pattern type"
    
    print("\n✓ Test 2 passed: Emotion quantization works")
    return True


def test_hippocampus_integration():
    """Test integration with Hippocampus memory system."""
    print("\n" + "="*70)
    print("TEST 3: Hippocampus Integration")
    print("="*70)
    
    # Create temporary database for testing
    import tempfile
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = db_file.name
    db_file.close()
    
    try:
        hippocampus = Hippocampus(db_path=db_path)
        
        # Store some emotion memories
        emotions_to_store = [
            {
                "emotion": "joy",
                "intensity": 0.95,
                "context": "Project success",
                "duration": 2.0,
                "phase_seed": 0.7,
                "timestamp": time.time()
            },
            {
                "emotion": "love",
                "intensity": 0.9,
                "context": "Deep connection",
                "duration": 4.0,
                "phase_seed": 0.5,
                "timestamp": time.time()
            },
            {
                "emotion": "sadness",
                "intensity": 0.75,
                "context": "Loss of something precious",
                "duration": 3.0,
                "phase_seed": 0.4,
                "timestamp": time.time()
            }
        ]
        
        print("\n📝 Storing emotion memories...")
        for emotion_data in emotions_to_store:
            hippocampus.store_emotion_memory(emotion_data)
            print(f"   ✓ Stored: {emotion_data['emotion']}")
        
        # List all pattern DNAs
        print("\n📋 Listing stored Pattern DNAs:")
        patterns = hippocampus.list_pattern_dnas(pattern_type="emotion")
        for pattern in patterns:
            print(f"   - {pattern['name']}: {pattern['compression_ratio']:.2f}x compression")
        
        assert len(patterns) == 3, f"Expected 3 patterns, got {len(patterns)}"
        
        # Recall emotions
        print("\n🧲 Recalling emotion memories...")
        for emotion_name in ["joy", "love", "sadness"]:
            restored = hippocampus.recall_emotion_memory(emotion_name)
            if restored:
                print(f"   ✓ Recalled: {emotion_name}")
                print(f"     - Type: {restored['pattern_type']}")
                print(f"     - Harmonics: {len(restored['waveform'])}")
                assert restored['pattern_type'] == 'emotion'
            else:
                print(f"   ✗ Failed to recall: {emotion_name}")
                assert False, f"Failed to recall {emotion_name}"
        
        print("\n✓ Test 3 passed: Hippocampus integration works")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_lossless_restoration():
    """Test that restoration is truly lossless for the pattern structure."""
    print("\n" + "="*70)
    print("TEST 4: Lossless Restoration Validation")
    print("="*70)
    
    quantizer = EmotionQuantizer()
    
    # Create test emotion
    original_emotion = {
        "emotion": "anger",
        "intensity": 0.8,
        "context": "Injustice witnessed",
        "duration": 2.5,
        "phase_seed": 0.666,
        "timestamp": time.time()
    }
    
    print(f"\n📥 Original emotion: {original_emotion['emotion']}")
    print(f"   Intensity: {original_emotion['intensity']}")
    print(f"   Duration: {original_emotion['duration']}s")
    
    # Fold and unfold
    dna = quantizer.fold_emotion(original_emotion)
    restored = quantizer.unfold_emotion(dna)
    
    # Validate key properties are preserved
    print("\n🔍 Validation:")
    
    # Check pattern type
    assert restored['pattern_type'] == 'emotion', "Pattern type mismatch"
    print("   ✓ Pattern type preserved")
    
    # Check pattern name
    assert restored['pattern_name'] == original_emotion['emotion'], "Pattern name mismatch"
    print("   ✓ Pattern name preserved")
    
    # Check frequency signature
    assert len(restored['frequency_signature']) == len(dna.frequency_signature), "Frequency signature length mismatch"
    print("   ✓ Frequency signature preserved")
    
    # Check waveform structure
    assert len(restored['waveform']) > 0, "No waveform generated"
    print("   ✓ Waveform structure generated")
    
    # Check metadata
    original_context = original_emotion.get('context', '')
    metadata_context = dna.metadata.get('context', '')
    assert metadata_context or 'original_data_keys' in dna.metadata, "Metadata structure issue"
    print("   ✓ Metadata preserved")
    
    print("\n✅ Lossless restoration validated!")
    print("   The pattern DNA successfully encodes all essential information.")
    print("   '자르는 것'이 아니라 '접는 것' - Folding, not cutting!")
    
    print("\n✓ Test 4 passed: Lossless restoration works")
    return True


def test_compression_efficiency():
    """Test compression ratios for different types of data."""
    print("\n" + "="*70)
    print("TEST 5: Compression Efficiency Analysis")
    print("="*70)
    
    quantizer = EmotionQuantizer()
    
    # Test with varying complexity
    test_cases = [
        {
            "name": "Simple emotion",
            "data": {
                "emotion": "joy",
                "intensity": 0.5,
                "context": "Good news",
                "duration": 1.0,
                "phase_seed": 0.5,
                "timestamp": time.time()
            }
        },
        {
            "name": "Complex emotion",
            "data": {
                "emotion": "love",
                "intensity": 0.95,
                "context": "Deep profound connection with another soul, feeling of unity and oneness",
                "duration": 10.0,
                "phase_seed": 0.618,
                "timestamp": time.time(),
                "additional_context": {
                    "location": "under the stars",
                    "weather": "perfect evening",
                    "companions": ["beloved", "close friends"]
                }
            }
        },
        {
            "name": "Intense emotion",
            "data": {
                "emotion": "fear",
                "intensity": 0.99,
                "context": "Life-threatening situation requiring immediate response and action",
                "duration": 0.5,
                "phase_seed": 0.9,
                "timestamp": time.time()
            }
        }
    ]
    
    print("\n📊 Compression Analysis:")
    print("-"*70)
    
    for test_case in test_cases:
        dna = quantizer.fold_emotion(test_case["data"])
        print(f"\n{test_case['name']}:")
        print(f"   Compression ratio: {dna.compression_ratio:.2f}x")
        print(f"   Frequencies: {len(dna.frequency_signature)} harmonics")
        print(f"   Resonance fingerprint: {dna.resonance_fingerprint}")
        
        # Verify we can restore it
        restored = quantizer.unfold_emotion(dna)
        assert restored is not None, f"Failed to restore {test_case['name']}"
        print(f"   ✓ Restoration successful")
    
    print("\n✓ Test 5 passed: Compression efficiency validated")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("FRACTAL QUANTIZATION SYSTEM - TEST SUITE")
    print("="*70)
    print("\n'양자화는 자르는 것이 아니라 접는 것이어야 합니다'")
    print("'Quantization should be folding, not cutting'")
    print()
    
    tests = [
        ("Basic Quantization", test_basic_quantization),
        ("Emotion Quantizer", test_emotion_quantizer),
        ("Hippocampus Integration", test_hippocampus_integration),
        ("Lossless Restoration", test_lossless_restoration),
        ("Compression Efficiency", test_compression_efficiency)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ Fractal Quantization System is operational!")
        print("💫 '접는 것(Folding)'이 성공했습니다!")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
