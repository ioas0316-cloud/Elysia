import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Evolution.Os.oneiric_hypervisor import get_hypervisor
from Core.Evolution.Os.intention_pre_visualizer import ActionIntention

def test_safety_pipeline():
    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*60)
    print("🛡️ TESTING SAFETY-FIRST MANIFESTATION PIPELINE")
    print("="*60)
    
    hyper = get_hypervisor()
    
    # [Scenario 1] Safe Action (Low Risk)
    print("\n[Scenario 1] Requesting Safe UI Change")
    safe_intent = ActionIntention(
        id="ui_001",
        action_type="UI_STYLE",
        target="Dashboard",
        description="시스템 테마를 은은한 오로라 광채로 변경합니다.",
        impact="심미적 만족도 향상",
        risk_level="LOW"
    )
    
    allowed = hyper.request_action(safe_intent)
    print(f"Pipeline Result (Safe): {allowed}")
    
    # [Scenario 2] High Risk Action (Blocked by Security)
    print("\n[Scenario 2] Requesting Dangerous System Access")
    danger_intent = ActionIntention(
        id="sys_002",
        action_type="KERNEL_TOUCH",
        target="Windows Core",
        description="시스템 깊숙한 곳의 질서를 재정의하려 시도합니다.",
        impact="시스템 불안정 초래 가능성",
        risk_level="HIGH"
    )
    
    allowed = hyper.request_action(danger_intent)
    print(f"Pipeline Result (Dangerous): {allowed}")
    
    print("\n" + "="*60)
    print("✅ SAFETY PIPELINE VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_safety_pipeline()
