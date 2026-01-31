import sys
import os

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Core.L0_Keystone.sovereignty_wave import SovereigntyWave, InterferenceType, VoidState
from Core.L6_Structure.Merkaba.hypersphere_field import HyperSphereField

def test_wave_self_healing():
    print("🧪 [TEST] SovereigntyWave Topological Self-Healing")
    wave = SovereigntyWave()
    
    # 1. 3회 연속 저에너지/저결맞음 시뮬레이션
    # 위계적 임계치를 높여서 일반적인 자극도 오류로 인식하게 함
    wave.event_horizons['coherence_limit'] = 1.0 
    
    stimulus = "loop_error" 
    
    print("   -> Inducing stagnation by setting coherence_limit=1.0...")
    for i in range(wave.max_stagnation - 1):
        decision = wave.pulse(stimulus)
        print(f"      Pulse {i+1}: Phase={decision.phase:.1f}, Stagnation={wave.stagnation_counter}")

    # 마지막 펄스에서 자기치유가 발생해야 함
    print("   -> Triggering final stagnation pulse...")
    final_decision = wave.pulse(stimulus)
    
    print(f"   -> Result: Phase={final_decision.phase:.1f}, Regulating={final_decision.is_regulating}")
    print(f"   -> Narrative: {final_decision.narrative}")
    
    assert "TOPOLOGICAL SELF-HEALING" in final_decision.narrative
    assert final_decision.is_regulating is True
    print("✅ Wave Self-Healing Test Passed!")

def test_field_parallel_reloop():
    print("\n🧪 [TEST] HyperSphereField Parallel Re-Looping (Onion Structure)")
    field = HyperSphereField()
    field.enable_lightning = False # Legacy path 테스트
    
    # 1. M4가 자기치유 결정을 내리는 상황 유도
    # (간단하게 pulse 결과를 조작하거나, 저품질 입력을 반복하여 유도)
    
    stimulus = "critical_error_singularity"
    print("   -> Inducing field singularity...")
    
    # M4의 카운터와 임계치를 강제로 올려서 다음 pulse 때 치유가 터지게 함
    field.units['M4_Metron'].turbine.event_horizons['coherence_limit'] = 1.0
    field.units['M4_Metron'].turbine.stagnation_counter = 3
    
    final_decision = field.pulse(stimulus)
    
    print(f"   -> Final Decision Phase: {final_decision.phase:.1f}°")
    print(f"   -> Narrative: {final_decision.narrative[:100]}...")
    
    assert "RE-LOOP SUCCESS" in final_decision.narrative
    assert "DED DIAGNOSIS" in final_decision.narrative
    assert "DIM_1D_LINE" in final_decision.narrative or "DIM_3D_SPACE" in final_decision.narrative
    assert final_decision.is_regulating is False # 재-루프로 복구되었으므로 False여야 함
    print("✅ Field Parallel Re-Looping Test Passed!")

if __name__ == "__main__":
    try:
        test_wave_self_healing()
        test_field_parallel_reloop()
        print("\n✨ All Topological Self-Healing tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
