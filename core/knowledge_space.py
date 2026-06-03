from typing import List, Dict
from core.rotor_gate import ConceptWave, RotorGate

class PhaseSpace:
    """
    A continuous space where concepts exist not in static hierarchies, 
    but as dynamic waves interacting with each other.
    """
    def __init__(self):
        self.concepts: Dict[str, ConceptWave] = {}
        self.rotor_gate = RotorGate()

    def add_concept(self, concept: ConceptWave):
        self.concepts[concept.name] = concept

    def introduce_criteria(self, concept_name_a: str, concept_name_b: str, axis: str):
        """
        Forces an interaction through the RotorGate.
        """
        a = self.concepts[concept_name_a]
        b = self.concepts[concept_name_b]
        
        # Ensure both concepts have the axis
        if a.get_phase(axis) is None:
            a.add_axis(axis, 0.0)
        if b.get_phase(axis) is None:
            b.add_axis(axis, 0.0)

        result = self.rotor_gate.interact(a, b, axis)
        return result

    def branch_differentiation(self, concept_name_a: str, concept_name_b: str, base_axis: str, new_axis: str):
        """
        Introduces a 'difference' criteria that forces a dimensional split.
        """
        a = self.concepts[concept_name_a]
        b = self.concepts[concept_name_b]
        result = self.rotor_gate.differentiate_new_axis(a, b, base_axis, new_axis)
        return result

    def auto_resonate(self, axis: str, tolerance: float = 0.05):
        """
        [Legacy] 하드코딩된 tolerance를 사용하는 이전 버전의 공명. 
        이제 discover_boundaries로 대체됩니다.
        """
        pass

    def discover_boundaries(self, axis: str) -> List[str]:
        """
        하드코딩된 기준값(Tolerance) 없이, 데이터 밀도를 스캔하여 스스로 경계(Boundary)를 찾아냅니다.
        간격이 급격히 넓어지는 구간(Event Horizon)을 다름의 경계로 선언합니다.
        """
        logs = []
        # 위상값을 기준으로 정렬
        sorted_concepts = sorted(
            [c for c in self.concepts.values() if c.get_phase(axis) is not None],
            key=lambda x: x.get_phase(axis)
        )
        
        if len(sorted_concepts) < 2:
            return logs

        # 1. 공간 내의 평균 밀도(간격) 계산
        gaps = []
        for i in range(len(sorted_concepts)-1):
            gap = sorted_concepts[i+1].get_phase(axis) - sorted_concepts[i].get_phase(axis)
            gaps.append(gap)
            
        avg_gap = sum(gaps) / len(gaps)
        
        # 2. 평균 간격보다 N배(예: 3배) 넓은 곳을 '경계선(Event Horizon)'으로 인식
        clusters = []
        current_cluster = [sorted_concepts[0]]
        
        for i in range(len(sorted_concepts)-1):
            gap = gaps[i]
            # 밀도 기반 경계 탐색 (평균 밀도와의 이탈도)
            if gap > avg_gap * 2.0:
                logs.append(f"Boundary Detected! Event Horizon established at Phase Gap: {gap:.4f} (Between [{sorted_concepts[i].name}] and [{sorted_concepts[i+1].name}])")
                clusters.append(current_cluster)
                current_cluster = [sorted_concepts[i+1]]
            else:
                current_cluster.append(sorted_concepts[i+1])
        
        clusters.append(current_cluster)

        # 3. 경계 내부에 있는 개체들은 공명(Velocity 부여), 경계가 나뉜 개체들은 직교 분화
        for cluster in clusters:
            cluster_names = [c.name for c in cluster]
            logs.append(f"Cluster formed: {cluster_names} resonating within its own boundary.")
            for c in cluster:
                c.apply_force(axis, 1.0) # 군집 내부 공명
                
            # 다른 군집과는 다름의 축을 생성 (분화)
            for other_cluster in clusters:
                if cluster != other_cluster:
                    for a in cluster:
                        for b in other_cluster:
                            new_axis = f"{axis}_Diff_{a.name}_{b.name}"
                            self.rotor_gate.differentiate_new_axis(a, b, axis, new_axis)
                            
        return logs

    def discover_boundaries_with_events(self, axis: str):
        """
        삼중 토러스용 확장: 물리적 경계를 찾으면서 '과정'을 EventWave 객체로 방출(Emit)합니다.
        """
        logs = self.discover_boundaries(axis)
        
        # 내부 구조 상 clusters를 다시 계산하는 것은 비효율적이나 시뮬레이션 목적상 다시 단순 계산
        sorted_concepts = sorted([c for c in self.concepts.values() if c.get_phase(axis) is not None], key=lambda x: x.get_phase(axis))
        if len(sorted_concepts) < 2:
            return logs, []

        gaps = [sorted_concepts[i+1].get_phase(axis) - sorted_concepts[i].get_phase(axis) for i in range(len(sorted_concepts)-1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 1.0
        
        clusters = []
        current_cluster = [sorted_concepts[0]]
        for i in range(len(sorted_concepts)-1):
            if gaps[i] > avg_gap * 2.0:
                clusters.append(current_cluster)
                current_cluster = [sorted_concepts[i+1]]
            else:
                current_cluster.append(sorted_concepts[i+1])
        clusters.append(current_cluster)

        from core.meta_perception import EventWave
        events = []
        for cluster in clusters:
            if len(cluster) >= 2:
                for i in range(len(cluster)-1):
                    # 공명(Resonance) 사건을 파동으로 객체화
                    ew = EventWave("Resonance", cluster[i].name, cluster[i+1].name, energy=1.0)
                    events.append(ew)
                    
            # 분화(Differentiation) 사건 객체화 (타 군집과의 분리)
            for other_cluster in clusters:
                if cluster != other_cluster:
                    ew_diff = EventWave("Differentiation", cluster[0].name, other_cluster[0].name, energy=gaps[0])
                    events.append(ew_diff)
                    break # 대표로 1개씩만 생성 (시뮬레이션 단순화)

        return logs, events

    def cross_dimensionalize(self, target_concept_name: str, human_label: str):
        """
        자율적으로 형성된 물리적 군집 경계 안으로 인간의 라벨(개념)이 유입되었을 때,
        해당 경계 내에 존재하는 모든 개체들에게 라벨을 '교차차원(Cross-Dimension)'으로 꽂아 넣습니다.
        """
        target = self.concepts.get(target_concept_name)
        if not target:
            return "Target not found."
            
        # 가장 지배적인 물리적 축(예: Natural_Entropy) 확인
        axis = "Natural_Entropy"
        target_phase = target.get_phase(axis)
        
        if target_phase is None:
            return "Target has no natural entropy phase."
            
        entangled = []
        
        # 앞서 discover_boundaries에서 형성된 '장력/운동성'을 기반으로 같은 군집 탐색
        # (간단한 구현을 위해 여기서는 대상과 매우 가까운 위상을 가진 개체를 군집으로 간주)
        for name, concept in self.concepts.items():
            p = concept.get_phase(axis)
            if p is not None:
                # 여기서 0.15 같은 하드코딩 대신, concept.velocity에 동일 축 추진력이 있는지로 확인
                # 하지만 일단 위상 근접도로 군집 내부임을 판별 (Event Horizon 내부)
                # 실제로는 clusters 배열을 멤버변수로 저장하여 꺼내쓰는 것이 정확함
                if abs(p - target_phase) < 0.2: # (Simulation simplification)
                    concept.add_axis(f"Label_{human_label}", 0.0)
                    concept.apply_force(f"Label_{human_label}", 10.0) # 라벨의 거대한 중력 획득
                    entangled.append(name)
                    
        return f"Cross-Dimensionalization! The human label '[Label_{human_label}]' has pierced through the physical boundary containing: {entangled}."

    def time_step(self):
        """
        시간의 흐름에 따라, 능동적 프랙탈 셀(Active Operator)들이 주변 공간을 자율 변형시키는 동력 엔진.
        (분류가 아니라 실시간 '생성'과 '이동'을 담당)
        """
        from core.meta_perception import ActiveFractalCell
        logs = []
        for name, concept in self.concepts.items():
            if isinstance(concept, ActiveFractalCell):
                res = concept.exert_influence(self.concepts)
                logs.extend(res)
        return logs

    def print_state(self):
        print("\n--- Current Phase Space State ---")
        for concept in self.concepts.values():
            print(concept)
        print("---------------------------------\n")
