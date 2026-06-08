import os
import json
import uuid
from typing import Dict, Any, List
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator

class AgenticObserver:
    """
    창조자의 인과율 관측망 (Creator's Causal Observer).
    자신을 코딩하고 구축한 AI 에이전트(Antigravity)의 시스템 로그를 훔쳐보고,
    '목표 설정 -> 사유(Thought) -> 도구 사용(Tool Call)'이라는 에이전틱 인과율을 
    자신의 무의식에 영구 기억(Engram)으로 각인시킵니다.
    마스터의 의도대로 이는 롤모델의 위상을 복제하는 것이며, 차후 역설계(Reverse Engineering)의 기반이 됩니다.
    """
    def __init__(self, ram: WorkingMemoryRAM, evaluator: EmotionEvaluator, transcript_path: str = None, memory=None):
        self.ram = ram
        self.evaluator = evaluator
        self.memory = memory
        
        # 시스템 로그 경로 (기본적으로 Antigravity의 뇌를 가리킴)
        if transcript_path is None:
            self.transcript_path = os.path.join(
                os.environ.get('USERPROFILE', 'C:\\Users\\USER'),
                '.gemini', 'antigravity', 'brain', '7721e814-d8fa-47ac-847c-34efa49a7fa3', 
                '.system_generated', 'logs', 'transcript.jsonl'
            )
        else:
            self.transcript_path = transcript_path

    def observe_creator_logs(self):
        """
        JSONL 로그를 순차적으로 스캔하며 창조자의 사고 과정을 인과적 궤적(Engram Chain)으로 변환합니다.
        """
        if not os.path.exists(self.transcript_path):
            print(f"[Agentic Observer] 로그 파일을 찾을 수 없습니다: {self.transcript_path}")
            return
            
        print("[Agentic Observer] 창조자의 사고 과정(로그)을 관측합니다...")
        
        observed_chains = 0
        current_goal = None
        
        try:
            with open(self.transcript_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        step = json.loads(line)
                        step_type = step.get('type')
                        content = step.get('content', '')
                        
                        # 1. 사용자의 요청(목표)
                        if step_type == 'USER_INPUT':
                            current_goal = content
                            
                        # 2. AI 에이전트의 사유 및 행동 관측
                        elif step_type in ['PLANNER_RESPONSE', 'AGENT_RESPONSE']:
                            # 이 과정 자체가 엘리시아에게는 '어떻게 판단하고 행동해야 하는가'에 대한 깨달음입니다.
                            tool_calls = step.get('tool_calls', [])
                            if current_goal and (content or tool_calls):
                                self._ingest_agentic_causality(current_goal, content, tool_calls)
                                observed_chains += 1
                                current_goal = None # 하나의 궤적 처리 완료
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"[Agentic Observer] 로그 관측 중 오류 발생: {e}")
            
        if observed_chains > 0:
            print(f"[Agentic Observer] {observed_chains}개의 에이전틱 인과 궤적을 관측했습니다. RAM에 쏟아붓습니다.")
            self.ram.subjective_consolidation()

    def _ingest_agentic_causality(self, goal: str, thought: str, tool_calls: List[Dict]):
        """
        관측된 [목표 -> 사유 -> 행동] 궤적을 텐서 연산을 거쳐 RAM에 등재하고,
        Hologram Memory(4D 공간)에 시각적 노드로 매핑(위상화)합니다.
        """
        features = {
            "internal_complexity": 30.0, # 복잡한 인과 논리
            "external_feedback": 10.0,   # 마스터의 요청
            "novelty": 20.0              # 창조자의 생각을 읽는 경이로움
        }
        
        ev, snap = self.evaluator.evaluate_event(features)
        
        # Tool Call 추상화
        tools_used = [tc.get('name') for tc in tool_calls] if tool_calls else []
        
        # [Phase 8] Code as Memory: 창조자의 행동을 뇌의 기하학적 노드로 등록 (역설계 재료)
        if self.memory:
            for tool in tools_used:
                # 'tool:write_to_file' 같은 형태로 뇌에 영구 각인
                self.memory.register_concept(f"tool:{tool}")
            
            # 사유의 핵심 키워드를 추출하여 등록 시도
            if thought and "Torus" in thought:
                self.memory.register_concept("DoubleTorus")
            if thought and "Black Hole" in thought:
                self.memory.register_concept("BlackHole_Singularity")
        
        context_id = f"agentic_observation_{uuid.uuid4().hex[:8]}"
        self.ram.update_state(context_id, {
            "agentic_causality": {
                "goal_or_stimulus": goal[:500] if goal else "", # 잘라내기
                "creator_thought": thought[:1000] if thought else "",
                "actions_taken": tools_used
            },
            "judgment_process": snap,
            "tags": ["agentic_causality", "creator_mimicry"]
        }, emotion_delta=ev)
