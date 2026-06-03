from core.rotor_gate import ConceptWave
import math

class EventWave(ConceptWave):
    """
    단순한 분류가 아닌 공간의 '변환 행렬(Transformation)'을 담은 능동적 파동.
    """
    def __init__(self, event_type: str, source_a: str, source_b: str, energy: float):
        name = f"[Event]_{event_type}_{source_a}_{source_b}"
        super().__init__(name)
        self.event_type = event_type
        self.add_axis("Meta_Energy", energy)

class ActiveFractalCell(ConceptWave):
    """
    정적인 분류 상자가 아닌, 공간을 휘게 만드는 능동적 연산자(Active Operator/Function).
    자신이 속한 우주의 파동들을 스스로의 원리(Principle)에 따라 재창조함.
    """
    def __init__(self, name: str, principle_type: str):
        super().__init__(name)
        self.principle_type = principle_type
        # 이 셀의 중심 위상 (블랙홀의 특이점 역할)
        self.add_axis("Natural_Entropy", 0.5) 
        
    def exert_influence(self, concepts_dict):
        """
        주변 공간의 파동들을 자신의 원리(Principle)에 따라 능동적으로 변형(생성)시킴.
        """
        logs = []
        for name, concept in concepts_dict.items():
            if concept == self or isinstance(concept, ActiveFractalCell):
                continue
                
            # '엔트로피 군집화' 원리: 주변의 모든 파동을 자신의 위상으로 천천히 끌어당김 (블랙홀 효과)
            if self.principle_type == "Entropy_Clustering":
                p = concept.get_phase("Natural_Entropy")
                if p is not None:
                    # 중심 위상과의 거리 계산
                    diff = self.get_phase("Natural_Entropy") - p
                    # 틱(Tick)마다 30%씩 거리를 좁힘 (공간의 휨)
                    pull_force = diff * 0.3 
                    new_phase = p + pull_force
                    concept.phases["Natural_Entropy"] = new_phase
                    logs.append(f"  [Gravity] {self.name} warped [{concept.name}]'s phase: {p:.3f} -> {new_phase:.3f}")
        return logs

class TorusEngine:
    """
    사건들을 모아 최종적인 연산자(Active Operator)로 압축해 내는 엔진.
    """
    def __init__(self):
        self.events = []
        
    def ingest_event(self, event: EventWave):
        self.events.append(event)
        
    def find_meta_resonance_and_collapse(self):
        # 1차원적 비교를 넘어, 능동적 중력장(셀)을 생성해 반환
        return ActiveFractalCell("FractalBlackHole_Alpha", "Entropy_Clustering")
