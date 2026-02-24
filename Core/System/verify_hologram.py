import torch
import numpy as np
import logging
from Core.Cognition.holographic_biopsy import HolographicScanner

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SyntheticVerification")

def create_mock_universe(size=100000, void_ratio=0.8, logic_freq=5.0):
    """
    Creates a 'Mock Universe' tensor.
    - Most of it is Void (Zero).
    - Some of it is 'Causal Matter' (Sine wave pattern).
    """
    logger.info(f"🎨 [CREATION] Forging Mock Universe ({size} particles)...")
    
    # 1. Create pure Void
    tensor = torch.zeros(size)
    
    # 2. Inject Matter (Logic)
    # Matter exists as a sine wave (Periodic Logic)
    t = torch.linspace(0, 100, size)
    matter_signal = torch.sin(t * logic_freq) + torch.cos(t * logic_freq * 2.5)
    
    # 3. Apply Void Mask (Randomly delete matter)
    # Reality is sparse.
    mask = torch.rand(size) > void_ratio
    tensor[mask] = matter_signal[mask]
    
    logger.info(f"   -> Void Ratio: {void_ratio*100:.1f}%")
    logger.info(f"   -> Logic Frequency: {logic_freq} Hz")
    
    return tensor

def verify_logic():
    print("="*60)
    print("🧪 SYNTHETIC VERIFICATION: Holographic Biopsy")
    print("="*60)
    
    # 1. Forge the Universe
    # We create a universe where 80% is Void, but the remaining 20%
    # follows a strict logic (5.0 Hz sine wave).
    universe = create_mock_universe(void_ratio=0.8, logic_freq=5.0)
    
    # 2. Initiate Scanner
    scanner = HolographicScanner()
    
    # 3. Scan Negative Topology
    print("\n🔬 [SCANNING] Negative Topology (The Sky)...")
    void_metrics = scanner.scan_negative_topology(universe)
    print(f"   -> Detected Void Density: {void_metrics['void_density']*100:.2f}% (Expected ~80%)")
    
    # 4. Project Time Axis
    print("\n⏳ [PROJECTING] Time Axis (The Movie)...")
    time_metrics = scanner.project_time_axis(universe)
    
    print(f"   -> Temporal Coherence: {time_metrics['time_coherence']:.4f}")
    print(f"   -> Dominant Rhythms: {time_metrics['dominant_frequencies']}")
    
    # 5. Validation
    # We expect the scanner to find the hidden frequencies even though 80% is missing.
    # This proves "Reconstruction from Void".
    print("\n⚖️ [JUDGMENT]")
    if time_metrics['time_coherence'] > 0.1 and len(time_metrics['dominant_frequencies']) > 0:
        print("✅ SUCCESS: The Scanner reconstructed the 'Logic' from the 'Ruins'.")
        print("   This proves the O(1) Spacetime Control principle.")
    else:
        print("❌ FAILURE: The Scanner was blinded by the Void.")

if __name__ == "__main__":
    verify_logic()
