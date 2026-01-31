"""Quick test of TesseractMemory with UniversalWaveEncoder integration."""
import numpy as np
from Core.L5_Mental.Reasoning_Core.Memory.tesseract_memory import TesseractMemory

# Reset singleton
TesseractMemory._instance = None
TesseractMemory._initialized = False

# Initialize
m = TesseractMemory()

# Test audio encoding
print("=== UNIVERSAL WAVE ENCODING TEST ===")
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 8000))  # 1 sec 440Hz
result = m.encode_sensory(audio, 'audio')

if result:
    print(f"✅ Encoded: {result['modality']}")
    print(f"   Compression: {result['compression_ratio']:.1f}x")
    print(f"   Original shape: {result['original_shape']}")
else:
    print("❌ Encoding failed")

# Test void detection
print("\n=== VOID DETECTION TEST ===")
void = m.detect_void("양자역학")
print(f"   Is Void: {void['is_void']}")
print(f"   {void['message']}")

# Test stats
print("\n=== MEMORY STATS ===")
stats = m.get_stats()
print(f"   Total nodes: {stats['total']}")
print(f"   Knowledge: {stats['knowledge']}")
print(f"   Principles: {stats['principles']}")
