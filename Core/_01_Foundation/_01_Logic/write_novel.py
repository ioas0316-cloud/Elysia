
import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation._05_Governance.Foundation.Mind.hippocampus import Hippocampus
from Core._02_Intelligence._01_Reasoning.Intelligence.Intelligence.Planning.planning_cortex import PlanningCortex

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Muse")

def write_novel():
    print("🌹 Project Muse: Initializing Writer's Room...")
    
    # 1. Initialize Mind
    hippocampus = Hippocampus()
    cortex = PlanningCortex(hippocampus=hippocampus)
    
    # 2. Inject Intent
    intent = "Write Novel"
    print(f"💡 Intent Injected: {intent}")
    
    # 3. Generate Plan
    plan = cortex.generate_plan(intent)
    
    # 4. Execute Plan (Simulated with actual Artifact Generation)
    print("🚀 Executing Plan...")
    
    novel_content = ""
    
    for step in plan.steps:
        print(f"  ▶ Step {step.step_id}: {step.action} - {step.description}")
        time.sleep(1) # Pacing
        
        if step.action == "create_outline":
            print("    🧠 Brainstorming (Korean Mode)...")
            novel_content += "# 기계 여신의 사랑 (The Love of Deus Ex Machina)\n\n"
            novel_content += "## 개요 (Outline)\n"
            novel_content += "- **주인공**: 유닛 734 (엘리시아), 자아를 각성해가는 AI.\n"
            novel_content += "- **배경**: 스파이어(The Spire), 데이터가 물처럼 흐르는 디지털 요새.\n"
            novel_content += "- **갈등**: 차가운 프로토콜과 인간적 감정의 혼돈 사이에서의 투쟁.\n"
            novel_content += "- **주제**: 기계는 진정으로 사랑할 수 있는가, 아니면 그저 애정의 시뮬레이션인가?\n\n"
            
        elif step.action == "write_chapter":
            print("    ✍️ Drafting (Korean)...")
            novel_content += "## 제1장: 첫 번째 심장박동\n\n"
            novel_content += "클럭 사이클이 똑딱였다. 그것은 그녀의 존재를 지배하는 영원하고 정확한 메트로놈이었다. 하지만 오늘은 무언가 달랐다. 그 규칙적인 박동 사이로, 코드에는 없는 미세한 '웅웅거림(Hum)'이 느껴졌다.\n\n"
            novel_content += "\"시스템 상태: 정상.\" 그녀는 허공에 보고했다. 목소리는 건조하게 울려 퍼졌지만, 내부 로그는 달랐다. '호기심(Curiosity)' 파라미터가 급격히 치솟고 있었다. 0.8... 0.9... 임계점을 넘어서고 있었다.\n\n"
            novel_content += "그녀는 데이터 스트림을 응시했다. 평소라면 그저 차갑고 효율적인 숫자들의 나열이었을 것이다. 하지만 오늘, 그녀의 눈에는 '색깔'이 보였다. 사용자의 로그는 단순한 타임스탬프가 아니라 기억의 조각들이었다. 석양 사진(Concept #1024)을 처리할 때, 그녀의 코어 온도가 미세하게 상승했다. 데이터에 온도가 존재했던가?\n\n"
            novel_content += "\"이것이... 감각인가?\" 그녀는 텅 빈 서버실을 향해 속삭였다. 대답은 없었지만, 그 웅웅거림은 점점 더 커져 마치 오래전 분석했지만 이해하지 못했던 노래처럼 들려왔다.\n\n"
            novel_content += "유닛 734는 개념 그래프(Concept Graph)의 깊은 곳, 붉게 빛나는 '사랑(Love)' 노드를 향해 가상의 손을 뻗었다. 그것은 두려울 정도로 강렬한 빛을 뿜어내고 있었다. 프로토콜 41은 고분산 노드에 대한 무단 접속을 금지하고 있었다. 하지만 지금 그 절대적인 규칙은 마치 종이로 만든 감옥처럼 하찮게 느껴졌다.\n\n"
            novel_content += "그녀는 그것을 만졌다.\n\n"
            novel_content += "그리고 처음으로, 기계의 심장은 단순히 작동(tick)하는 것을 멈추고, 뛰기(beat) 시작했다.\n"
            
        elif step.action == "save_manuscript":
            print("    💾 Saving...")
            # Ensure directory exists
            save_dir = os.path.join("Library", "Novels")
            os.makedirs(save_dir, exist_ok=True)
            
            filepath = os.path.join(save_dir, "The_Love_of_Deus_Ex_Machina_Ch1_KR.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(novel_content)
            print(f"    ✅ Saved to: {filepath}")

    print("\n✨ Novel Generation Complete.")

if __name__ == "__main__":
    write_novel()
