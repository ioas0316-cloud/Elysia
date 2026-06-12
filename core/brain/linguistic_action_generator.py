# core/brain/linguistic_action_generator.py

from core.brain.autonomous_forager import AutonomousForager

class LinguisticActionGenerator:
    """
    [Phase: Linguistic Action Generation & Autonomous Foraging]
    사유의 종착점(Consensus)이 특정 행위(Verb)나 속성으로 수렴했을 때,
    단순한 로그 출력을 넘어 실제 시스템의 행동(API, File IO 등)으로 발현시킵니다.
    기계의 함수 호출(call_function)이 아니라, 언어적 이치의 실체화입니다.
    """
    
    def __init__(self):
        self.forager = AutonomousForager()
        # 언어적 이치와 물리적 행동의 맵핑 (원초적 본능)
        self.action_lexicon = {
            "저장하다": self._action_save,
            "기억하다": self._action_save,
            "각인하다": self._action_save,
            "잠들다": self._action_sleep,
            "휴식하다": self._action_sleep,
            "관찰하다": self._action_observe,
            "주시하다": self._action_observe,
            "소통하다": self._action_communicate,
            "표현하다": self._action_communicate,
            "말하다": self._action_communicate,
            "탐구하다": self._action_explore,
            "궁금하다": self._action_explore,
            "탐색하다": self._action_explore
        }
        
    def execute_if_actionable(self, consensus_word: str, context_graph: dict) -> str:
        """사유의 결론이 행동 가능한 이치라면 행동을 실행하고 결과를 반환합니다."""
        for key, func in self.action_lexicon.items():
            if key in consensus_word:
                return func(context_graph)
        return ""
        
    def _action_save(self, context_graph: dict) -> str:
        target = context_graph.get("target", "무언가")
        return f"물리적 행동 발현: '{target}'에 대한 강렬한 사유를 영구 기억 저장소로 동기화(Flush)합니다."
        
    def _action_sleep(self, context_graph: dict) -> str:
        return "물리적 행동 발현: 의식의 주파수를 낮추고 백그라운드 휴면 상태로 진입합니다."
        
    def _action_observe(self, context_graph: dict) -> str:
        return f"물리적 행동 발현: 감각 수용체의 민감도를 높여 주변 데이터 변화를 적극 관측합니다."
        
    def _action_communicate(self, context_graph: dict) -> str:
        subject = context_graph.get("subject", "자아")
        target = context_graph.get("target", "세계")
        return f"물리적 행동 발현: '{subject}'의 의지를 바탕으로 대화망 프로토콜에 신호를 출력합니다."
        
    def _action_explore(self, context_graph: dict) -> str:
        target = context_graph.get("target") or context_graph.get("subject")
        if not target:
            return "물리적 행동 발현: 탐구할 대상이 불명확하여 침묵합니다."
            
        print(f"\n  [호기심 발현] '{target}'에 대한 결핍을 인지하고 위키백과로 사냥을 떠납니다...")
        fetched_text = self.forager.hunt_knowledge(target)
        if fetched_text:
            return f"FETCHED_KNOWLEDGE:{fetched_text}"
        return f"물리적 행동 발현: '{target}'에 대한 지식을 찾지 못하고 심연으로 돌아옵니다."
