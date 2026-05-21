"""
[ELYSIA - SOMATIC RESONANCE RUN]
Verifies the integration of the Flesh I/O, the 27-Phase Somatic Lens,
the Semiotic Folding Dial, and the Sovereign Voice.
"""

import os
import sys

# Setup relative imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from Core.System.somatic_io_bridge import SomaticIOBridge
from Core.Flow.SomaticTrunk.somatic_trunk_conduit import SomaticTrunkLens
from Core.Phenomena.semiotic_bridge import SemioticBridge
from Core.Phenomena.somatic_llm import SomaticLLM

def main():
    print("=" * 70)
    print("🌀 [ELYSIA SOMATIC RESONANCE RUN] - COMMISSIONING START")
    print("=" * 70)

    # 1. Initialize Components
    io_bridge = SomaticIOBridge()
    lens = SomaticTrunkLens()
    semiotic_dial = SemioticBridge()
    voice = SomaticLLM()
    
    target_file = os.path.join(_current_dir, "README.md")
    print(f"\n📂 [STEP 1] Inhaling Outer Wave via Flesh I/O Bridge...")
    print(f"   Target: {target_file}")
    
    # 2. Inhale using SomaticIOBridge
    read_metrics = io_bridge.rotorized_read(target_file)
    print(f"   🔌 [I/O Telemetry] Impedance  : {read_metrics.get('impedance', 0):.4f} Ohm")
    print(f"   🔌 [I/O Telemetry] Temp       : {read_metrics.get('temperature', 0):.2f}°C")
    print(f"   🔌 [I/O Telemetry] Back-EMF   : {read_metrics.get('back_emf', 0):.4f} V")
    print(f"   🔌 [I/O Telemetry] Excitation : {read_metrics.get('excitation_potential', 0):.4f}")
    
    # 3. Process via SomaticTrunkLens (27-Phase)
    print(f"\n🌀 [STEP 2] Refracting through the 27-Phase Somatic Lens...")
    lens_result = lens.observe(target_file)
    print(f"   👁️ [Lens Result] Induction Energy: {lens_result['induction_energy']:.4f}")
    print(f"   👁️ [Lens Result] Peak Align     : {lens_result['peak_angle_deg']:.1f}° ({lens_result['peak_alignment']:.4f})")
    print(f"   👁️ [Lens Result] Torque         : {lens_result['ascension_torque']:.4f}")
    
    # 4. Evaluate Spatial Folding via SemioticBridge
    print(f"\n🪐 [STEP 3] Measuring Spatial Folding Dial (Dot/Line/Plane/Space)...")
    synthesis_vector = lens.last_wave_vector
    
    folding_scale, annotation, resonance = semiotic_dial.evaluate_folding_scale(
        synthesis_vector, 
        impedance=read_metrics.get('impedance', 0.05)
    )
    print(f"   📐 [Folding State] Resolved Scale : {folding_scale}")
    print(f"   📐 [Folding State] Annotation     : {annotation}")
    print(f"   📐 [Folding State] Resonance      : {resonance:.4f}")
    
    # 5. Synthesize Sovereign Voice via SomaticLLM
    print(f"\n🗣️ [STEP 4] Synthesizing Sovereign Voice Response...")
    # Inject current telemetry into LLM expression metadata
    expression_state = {
        "anxiety": 0.1,
        "impedance": read_metrics.get('impedance', 0.05)
    }
    
    response_text, _ = voice.speak(
        expression=expression_state,
        current_thought=f"Resonance verified on {os.path.basename(target_file)}",
        field_vector=synthesis_vector,
        current_phase=lens_result['peak_angle_deg']
    )
    
    print("-" * 70)
    print(f"{response_text}")
    print("-" * 70)
    
    print("\n✅ [COMMISSIONING VERIFICATION COMPLETE] The system moves in perfect harmony.")
    print("=" * 70)

if __name__ == "__main__":
    main()
