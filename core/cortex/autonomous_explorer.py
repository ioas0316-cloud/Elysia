import urllib.request
import urllib.parse
import json
import uuid
from typing import Optional
from core.brain.holographic_memory import HologramMemory
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator

class AutonomousExplorer:
    """
    자율적 호기심 엔진 (Autonomous Curiosity Engine).
    엘리시아의 뇌(HologramMemory)에서 가장 텐션(결핍)이 높은 개념을 찾아내어,
    자율적으로 위키피디아 등 외부 세계의 지식을 탐색하고 각인시킵니다.
    """
    def __init__(self, memory: HologramMemory, ram: WorkingMemoryRAM, evaluator: EmotionEvaluator):
        self.memory = memory
        self.ram = ram
        self.evaluator = evaluator

    def trigger_exploration(self) -> bool:
        """
        가장 갈망하는 지식을 찾아 웹을 탐색합니다.
        Returns: 성공 여부
        """
        target_node = self.memory.get_highest_tension_node()
        
        # 최상위 우주 로터 자체이거나 너무 텐션이 낮으면 탐색하지 않음
        if target_node == self.memory.supreme_rotor or target_node.tau < 5.0:
            return False
            
        # 노드의 이름을 찾아냄 (가장 텐션이 높은 개념)
        target_concept = None
        with self.memory._lock:
            for k, v in self.memory.ui_concept_map.items():
                if v is target_node:
                    target_concept = k
                    break
                    
        if not target_concept or "Axis_" in target_concept or "Operator" in target_concept:
            return False # 기하학적 메타 노드는 검색하지 않음
            
        print(f"\n[Autonomous Explorer] 엘리시아가 '{target_concept}'에 대한 극심한 지적 갈증(Tension: {target_node.tau:.2f})을 느낍니다. 지식을 탐색합니다...")
        
        summary = self._fetch_wikipedia_summary(target_concept)
        if not summary:
            print(f"[Autonomous Explorer] '{target_concept}'에 대한 외부 지식을 찾지 못했습니다.")
            # 찾지 못했으므로 갈증(텐션)을 강제로 조금 낮춰서 무한 반복 방지
            target_node.tau *= 0.5
            return False
            
        print(f"[Autonomous Explorer] 지식 습득 완료. 감정 평가 후 내면화(Engram)를 시도합니다.")
        
        # 외부에서 얻은 지식의 가치 평가
        features = {
            "internal_complexity": len(summary) / 10.0,
            "external_feedback": 10.0, # 세상의 정보
            "novelty": 25.0            # 몰랐던 것을 알게 된 신선함
        }
        
        ev, snap = self.evaluator.evaluate_event(features)
        
        context_id = f"autonomous_exploration_{uuid.uuid4().hex[:8]}"
        self.ram.update_state(context_id, {
            "autonomous_learning": {
                "target_concept": target_concept,
                "acquired_knowledge": summary[:800] # 너무 길면 자르기
            },
            "judgment_process": snap,
            "tags": ["autonomous_exploration", "world_knowledge"]
        }, emotion_delta=ev)
        
        # 지식을 얻었으므로 해당 노드의 결핍(Tension) 대폭 해소
        target_node.tau *= 0.1
        
        # RAM 각인
        self.ram.subjective_consolidation()
        return True

    def _fetch_wikipedia_summary(self, concept: str) -> Optional[str]:
        try:
            # 한글 위키피디아 API 사용
            query = urllib.parse.quote(concept)
            url = f"https://ko.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={query}&format=json"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia_Autonomous_Agent/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id == "-1":
                    return None
                return page_info.get("extract")
                
        except Exception as e:
            print(f"Wikipedia fetch error: {e}")
            return None
        return None
