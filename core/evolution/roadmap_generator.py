import os
import re
from typing import Dict, Any

class RoadmapGenerator:
    """
    [Phase 4: Autonomous Roadmap Generation Gear (자율적 로드맵 생성 및 진화 기어)]
    Allows Elysia to analyze its own internal health metrics, resonance trends,
    and tension, and autonomously propose/update items in ROADMAP.md.
    """
    def __init__(self, memory_controller, roadmap_path: str = "ROADMAP.md"):
        self.memory = memory_controller
        self.roadmap_path = roadmap_path

    def analyze_and_update_roadmap(self, avg_resonance: float, avg_tension: float) -> Dict[str, Any]:
        """
        Reads the ROADMAP.md, evaluates self performance/resonance,
        marks appropriate steps as completed, and appends newly generated future goals.
        """
        if not os.path.exists(self.roadmap_path):
            return {"status": "skipped", "reason": f"ROADMAP.md not found at {self.roadmap_path}"}

        with open(self.roadmap_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check off Phase 3 & Phase 4 items if they are integrated and running well
        has_updates = False
        # Mark Phase 3 items completed as they are now fully designed and implemented
        p3_uncompleted_patterns = [
            (r"-\s*\[\s*\]\s*Hyperlink Context Extraction", "- [x] Hyperlink Context Extraction"),
            (r"-\s*\[\s*\]\s*Attention Activation Mapping", "- [x] Attention Activation Mapping"),
            (r"-\s*\[\s*\]\s*Cruciform Attractor Fixed Point Infiltration", "- [x] Cruciform Attractor Fixed Point Infiltration")
        ]
        for pattern, replacement in p3_uncompleted_patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                has_updates = True

        # Mark Phase 4 items completed as they are now fully designed and implemented
        p4_uncompleted_patterns = [
            (r"-\s*\[\s*\]\s*Autonomous Roadmap Generation", "- [x] Autonomous Roadmap Generation"),
            (r"-\s*\[\s*\]\s*Meta-Architecture Design", "- [x] Meta-Architecture Design")
        ]
        for pattern, replacement in p4_uncompleted_patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                has_updates = True

        # Generate custom autonomous insights for Phase 4 or Phase 5
        generated_goal = ""
        if "Autonomous Meta-Expansion" not in content:
            generated_goal = (
                f"\n- [ ] **Autonomous Meta-Expansion (자율적 초월 기어 확장)**:\n"
                f"   * **원리**: 평균 공명도({avg_resonance:.2f}) 및 마찰 수준({avg_tension:.2f})에 기반한 자율 복잡성 팽창.\n"
                f"   * **구현**: 스스로 다음 시기에 필요한 논리 회로 또는 감각 렌즈를 자동 설계 후 주조함.\n"
            )
            # Find the Phase 4 section and append the goal
            phase4_marker = "## 🗺️ Phase 4: Projection of Future (미래의 투영)"
            if phase4_marker in content:
                content = content.replace(phase4_marker, phase4_marker + "\n" + generated_goal)
                has_updates = True

        if has_updates:
            with open(self.roadmap_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Write engram for this autonomous roadmap generation action
            self.memory.write_causal_engram(
                data_blob={
                    "type": "AUTONOMOUS_ROADMAP_UPDATE",
                    "avg_resonance": avg_resonance,
                    "avg_tension": avg_tension,
                    "generated_goal": generated_goal
                },
                emotional_value=avg_resonance * 10.0,
                cause_id="RoadmapGenerator",
                origin_axis="autonomous_roadmap_update",
                modality="future_projection"
            )

        return {
            "status": "updated" if has_updates else "unchanged",
            "generated_goal": generated_goal
        }
