import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Intelligence.Intelligence.integrated_cognition_system import get_integrated_cognition
from Core.Foundation.Wave.wave_tensor import WaveTensor

def test_maturity_calibration():
    print("Testing Phase 39: Cognitive Maturity Calibration...")
    mind = get_integrated_cognition()
    
    # 1. Low Maturity Test (Child-like/Direct)
    print("\n[Scenario 1] Low Maturity State")
    mind.maturity.maturity_score = 0.2
    raw_speech = "알겠어. 내가 이거 해볼게."
    calibrated = mind.maturity.calibrate_expression(raw_speech)
    print(f"Raw: {raw_speech}")
    print(f"Calibrated: {calibrated}")
    
    # 2. High Maturity Test (Adult/Nuanced)
    print("\n[Scenario 2] High Maturity State")
    mind.maturity.maturity_score = 0.8
    raw_speech = "알겠어. 내가 이거 해볼게."
    calibrated = mind.maturity.calibrate_expression(raw_speech)
    print(f"Raw: {raw_speech}")
    print(f"Calibrated: {calibrated}")
    
    # 3. Wave-based Evaluation Test
    print("\n[Scenario 3] Wave-based Resonance Evaluation")
    # Create a high-frequency adult wave (852Hz)
    adult_wave = WaveTensor("Adult Thought")
    adult_wave.add_component(852.0, 1.0)
    
    result = mind.maturity.evaluate_resonance(adult_wave)
    print(f"Thought Frequencies: {adult_wave.active_frequencies}")
    print(f"Maturity Result: {result}")
    
    # Create a low-frequency wave (100Hz)
    infant_wave = WaveTensor("Basic Thought")
    infant_wave.add_component(100.0, 1.0)
    result = mind.maturity.evaluate_resonance(infant_wave)
    print(f"\nThought Frequencies: {infant_wave.active_frequencies}")
    print(f"Maturity Result: {result}")

if __name__ == "__main__":
    test_maturity_calibration()
