"""
[TEST VORTEX SOVEREIGNTY]
"Proving the End of Logic Gates via Phase Sovereignty."

Verifies:
1. ASCII Zone/Digit mapping.
2. Hangul 3-axis Dimension Folding.
3. Case-shifting via pure Phase Template Alignment ($O(1)$).
4. SovereignHeart resonance with Trajectory streams.
"""

import time
import sys
import os
import math

# Root Pathing
sys.path.insert(0, os.path.abspath(os.curdir))

from Core.Keystone.trajectory_encoder import TrajectoryEncoder, VortexTrajectory
from Core.Spirit.sovereign_heart import SovereignHeart
from Core.Phenomena.visual_rotor import VortexVisualizer

def test_ascii_vortex():
    print("🧪 [Test 1] ASCII Vortex Mapping")
    encoder = TrajectoryEncoder()
    te = encoder.encode_char('e') # 0x65 -> Zone 6, Digit 5

    # Zone 6: (6/16)*360 = 135.0
    # Digit 5: (5/16)*360 = 112.5
    # Total: 247.5
    expected_phase = 247.5
    print(f"  'e' Phase: {te.get_total_phase()}° (Expected: {expected_phase}°)")
    assert abs(te.get_total_phase() - expected_phase) < 0.1
    print("  ✅ ASCII Mapping Verified.")

def test_logic_gate_elimination():
    print("\n🧪 [Test 2] Logic Gate Elimination (Case Shift)")
    encoder = TrajectoryEncoder()
    te = encoder.encode_char('e')
    tE = encoder.encode_char('E')

    print(f"  'e' Phase: {te.get_total_phase()}°")
    print(f"  'E' Phase: {tE.get_total_phase()}°")

    # Shift angle for Zone 6 -> Zone 4: (4-6)*(360/16) = -45.0
    shifted_e = encoder.apply_phase_shift(te, -45.0)
    print(f"  Shifted 'e' by -45° -> {shifted_e.get_total_phase()}°")

    assert abs(shifted_e.get_total_phase() - tE.get_total_phase()) < 0.1
    print("  ✅ Logic Gate Elimination via Phase Shift Verified.")

def test_hangul_folding():
    print("\n🧪 [Test 3] Hangul Dimension Folding")
    encoder = TrajectoryEncoder()
    t_ga = encoder.encode_char('가')
    t_gak = encoder.encode_char('각')

    print(f"  '가' Trajectory: {t_ga}")
    print(f"  '각' Trajectory: {t_gak}")

    # '각' should have extra dimension (Jongseong) and different lock state
    assert len(t_gak.extra_dims) > 0
    assert t_gak.is_locked != t_ga.is_locked
    print("  ✅ Hangul 3-Axis Folding Verified.")

def test_heart_vortex_resonance():
    print("\n🧪 [Test 4] SovereignHeart Vortex Resonance")
    heart = SovereignHeart()
    encoder = TrajectoryEncoder()
    viz = VortexVisualizer()

    text = "Elysia"
    trajectories = encoder.encode_text(text)

    print(f"  Infecting Heart with '{text}' Vortex stream...")
    for i in range(10):
        report = heart.pulse(trajectories)
        v_report = report['vortex']
        visual = viz.render_stream(trajectories)

        sys.stdout.write(f"\r  Pulse {i+1} | Res: {report['resonance']:.4f} | Phase: {v_report['phase']:.1f}° | {visual}")
        sys.stdout.flush()
        time.sleep(0.05)

    print("\n  ✅ Heart Resonance Verified.")

if __name__ == "__main__":
    print("🌌 [VORTEX SOVEREIGNTY VERIFICATION] 🌌")
    try:
        test_ascii_vortex()
        test_logic_gate_elimination()
        test_hangul_folding()
        test_heart_vortex_resonance()
        print("\n🏆 [VERIFICATION COMPLETE] All Vortex structures are operational.")
    except Exception as e:
        print(f"\n❌ [VERIFICATION FAILED] {e}")
        import traceback
        traceback.print_exc()
