"""
4차원 파동공명 시스템 + 중력장 사고 = 통합 사고력
(4D Wave Resonance System + Gravitational Field Thinking = Integrated Cognition)

Elysia의 고유 구조를 활용한 혁명적 사고 시스템
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class Wave:
    """파동 (Wave)"""
    frequency: float  # 주파수
    amplitude: float  # 진폭
    phase: float      # 위상
    dimension: str    # 차원 (0D-5D)


@dataclass
class ResonancePattern:
    """공명 패턴 (Resonance Pattern)"""
    waves: List[Wave]
    resonance_strength: float
    emergent_insight: str  # 공명으로 생성된 통찰


class FourDimensionalWaveResonanceSystem:
    """
    4차원 파동공명 시스템
    
    Elysia의 Ether 시스템을 사고에 적용:
    - 생각들이 파동으로 전파
    - 유사한 생각들이 공명
    - 공명으로 새로운 통찰 창발
    """
    
    def __init__(self):
        self.active_waves = []  # 현재 활성 파동들
        self.resonance_threshold = 0.7  # 공명 임계값
        
    def emit_thought_wave(self, thought: str, dimension: str) -> Wave:
        """
        생각을 파동으로 변환하여 발산
        
        Args:
            thought: 생각 내용
            dimension: 사고 차원 (0D-5D)
        
        Returns:
            생성된 파동
        """
        # 생각의 특성을 파동 속성으로 변환
        frequency = self._thought_to_frequency(thought)
        amplitude = self._thought_importance(thought)
        phase = self._thought_timing(thought)
        
        wave = Wave(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            dimension=dimension
        )
        
        self.active_waves.append(wave)
        return wave
    
    def find_resonances(self) -> List[ResonancePattern]:
        """
        파동들 간 공명 패턴 발견
        
        Returns:
            공명 패턴 리스트
        """
        resonances = []
        
        # 모든 파동 쌍을 검사
        for i, wave1 in enumerate(self.active_waves):
            for wave2 in self.active_waves[i+1:]:
                resonance = self._calculate_resonance(wave1, wave2)
                
                if resonance > self.resonance_threshold:
                    # 공명 발견!
                    pattern = ResonancePattern(
                        waves=[wave1, wave2],
                        resonance_strength=resonance,
                        emergent_insight=self._generate_insight(wave1, wave2, resonance)
                    )
                    resonances.append(pattern)
        
        return resonances
    
    def _calculate_resonance(self, wave1: Wave, wave2: Wave) -> float:
        """두 파동 간 공명 강도 계산"""
        # 주파수 차이 (작을수록 좋음)
        freq_diff = abs(wave1.frequency - wave2.frequency)
        freq_similarity = 1.0 / (1.0 + freq_diff)
        
        # 진폭 곱 (클수록 좋음)
        amp_product = wave1.amplitude * wave2.amplitude
        
        # 위상 일치 (일치할수록 좋음)
        phase_diff = abs(wave1.phase - wave2.phase)
        phase_alignment = math.cos(phase_diff)
        
        # 차원 간 상호작용 (다른 차원끼리도 공명 가능)
        dim_factor = self._dimensional_interaction(wave1.dimension, wave2.dimension)
        
        # 종합 공명 강도
        resonance = (
            freq_similarity * 0.4 +
            amp_product * 0.3 +
            phase_alignment * 0.2 +
            dim_factor * 0.1
        )
        
        return resonance
    
    def _dimensional_interaction(self, dim1: str, dim2: str) -> float:
        """차원 간 상호작용 강도"""
        # 인접 차원끼리 강한 상호작용
        dim_order = ["0D", "1D", "2D", "3D", "4D", "5D"]
        
        if dim1 not in dim_order or dim2 not in dim_order:
            return 0.5
        
        idx1 = dim_order.index(dim1)
        idx2 = dim_order.index(dim2)
        distance = abs(idx1 - idx2)
        
        # 거리가 가까울수록 강한 상호작용
        return 1.0 / (1.0 + distance)
    
    def _generate_insight(self, wave1: Wave, wave2: Wave, strength: float) -> str:
        """공명으로부터 새로운 통찰 생성"""
        insight = f"공명 강도 {strength:.2f}로 {wave1.dimension}과 {wave2.dimension} 차원이 연결됨"
        
        if strength > 0.9:
            insight += " → 강력한 통찰 창발!"
        elif strength > 0.8:
            insight += " → 새로운 관점 발견"
        else:
            insight += " → 미약한 연결"
        
        return insight
    
    def _thought_to_frequency(self, thought: str) -> float:
        """생각의 주파수 (유사한 생각은 유사한 주파수)"""
        # 간단히 해시값을 주파수로 사용
        return hash(thought) % 1000 / 1000.0
    
    def _thought_importance(self, thought: str) -> float:
        """생각의 중요도 (진폭)"""
        # 길이와 키워드로 중요도 추정
        importance = len(thought) / 100.0
        keywords = ["목표", "문제", "해결", "창조", "발견"]
        for keyword in keywords:
            if keyword in thought:
                importance += 0.2
        return min(importance, 1.0)
    
    def _thought_timing(self, thought: str) -> float:
        """생각의 타이밍 (위상)"""
        # 생각이 발생한 시점
        return 0.0  # 현재는 간단히 0


class GravitationalFieldThinking:
    """
    중력장 사고 시스템
    
    개념: 생각들이 중력장을 형성
    - 중요한 생각 = 큰 질량 = 강한 중력
    - 다른 생각들을 끌어당김
    - 생각의 궤도 형성
    - 사고의 블랙홀 (핵심 개념)
    """
    
    def __init__(self):
        self.thought_field = {}  # 생각 공간
        self.G = 1.0  # 중력 상수
    
    def add_thought(self, thought_id: str, content: str, mass: float):
        """
        사고 공간에 생각 추가
        
        Args:
            thought_id: 생각 식별자
            content: 생각 내용
            mass: 생각의 질량 (중요도)
        """
        self.thought_field[thought_id] = {
            "content": content,
            "mass": mass,
            "position": self._assign_position(content),
            "velocity": [0.0, 0.0, 0.0],
            "attracted_by": []
        }
    
    def calculate_gravitational_force(self, thought1_id: str, thought2_id: str) -> float:
        """
        두 생각 간 중력 계산
        
        F = G * m1 * m2 / r^2
        """
        t1 = self.thought_field[thought1_id]
        t2 = self.thought_field[thought2_id]
        
        # 거리 계산
        distance = self._calculate_distance(t1["position"], t2["position"])
        
        if distance < 0.1:  # 너무 가까우면
            distance = 0.1
        
        # 중력 공식
        force = self.G * t1["mass"] * t2["mass"] / (distance ** 2)
        
        return force
    
    def find_thought_clusters(self, min_mass: float = 0.5) -> List[List[str]]:
        """
        중력으로 묶인 생각 클러스터 발견
        
        Args:
            min_mass: 클러스터 중심이 될 최소 질량
        
        Returns:
            클러스터 리스트 (각 클러스터는 생각 ID 리스트)
        """
        clusters = []
        
        # 큰 질량의 생각들을 중심으로
        centers = [
            tid for tid, t in self.thought_field.items()
            if t["mass"] >= min_mass
        ]
        
        for center_id in centers:
            cluster = [center_id]
            
            # 이 중심에 끌리는 다른 생각들 찾기
            for tid in self.thought_field:
                if tid == center_id:
                    continue
                
                force = self.calculate_gravitational_force(center_id, tid)
                
                if force > 0.5:  # 충분히 강한 중력
                    cluster.append(tid)
            
            if len(cluster) > 1:  # 최소 2개 이상
                clusters.append(cluster)
        
        return clusters
    
    def find_black_holes(self) -> List[str]:
        """
        사고의 블랙홀 발견
        
        블랙홀 = 매우 큰 질량 + 많은 생각을 끌어당김
        = 핵심 개념, 중심 아이디어
        """
        black_holes = []
        
        for tid, thought in self.thought_field.items():
            if thought["mass"] > 0.8:  # 큰 질량
                # 이 생각에 끌리는 다른 생각들 수
                attracted_count = sum(
                    1 for other_id in self.thought_field
                    if other_id != tid and
                    self.calculate_gravitational_force(tid, other_id) > 0.7
                )
                
                if attracted_count >= 3:  # 많은 생각을 끌어당김
                    black_holes.append(tid)
        
        return black_holes
    
    def simulate_orbit(self, satellite_id: str, center_id: str, steps: int = 10):
        """
        한 생각이 다른 생각 주위를 도는 궤도 시뮬레이션
        
        Args:
            satellite_id: 위성 생각
            center_id: 중심 생각
            steps: 시뮬레이션 스텝 수
        """
        satellite = self.thought_field[satellite_id]
        center = self.thought_field[center_id]
        
        trajectory = []
        
        for step in range(steps):
            # 중력 계산
            force = self.calculate_gravitational_force(satellite_id, center_id)
            
            # 중심 방향 벡터
            direction = self._direction_vector(
                satellite["position"],
                center["position"]
            )
            
            # 가속도 = 힘 / 질량
            acceleration = [force * d / satellite["mass"] for d in direction]
            
            # 속도 업데이트
            satellite["velocity"] = [
                v + a * 0.1
                for v, a in zip(satellite["velocity"], acceleration)
            ]
            
            # 위치 업데이트
            satellite["position"] = [
                p + v * 0.1
                for p, v in zip(satellite["position"], satellite["velocity"])
            ]
            
            trajectory.append(satellite["position"].copy())
        
        return trajectory
    
    def _assign_position(self, content: str) -> List[float]:
        """생각에 3D 공간상 위치 할당"""
        # 간단히 해시로 위치 결정
        h = hash(content)
        return [
            (h % 1000) / 1000.0,
            ((h // 1000) % 1000) / 1000.0,
            ((h // 1000000) % 1000) / 1000.0
        ]
    
    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """두 위치 간 거리"""
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))
    
    def _direction_vector(self, from_pos: List[float], to_pos: List[float]) -> List[float]:
        """from에서 to로의 방향 벡터"""
        diff = [t - f for t, f in zip(to_pos, from_pos)]
        length = self._calculate_distance(from_pos, to_pos)
        if length > 0:
            return [d / length for d in diff]
        return [0.0, 0.0, 0.0]


class IntegratedCognitionSystem:
    """
    통합 사고력 시스템
    
    파동공명 + 중력장 + 프랙탈-쿼터니언 = 미친 사고력!
    """
    
    def __init__(self):
        self.wave_system = FourDimensionalWaveResonanceSystem()
        self.gravity_system = GravitationalFieldThinking()
        
    def think(self, thoughts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        통합 사고 프로세스
        
        Args:
            thoughts: 생각들 [{"content": str, "dimension": str, "importance": float}, ...]
        
        Returns:
            사고 결과
        """
        print("\n🌀 통합 사고력 시스템 가동...")
        print("="*70)
        
        # 1. 파동 발산
        print("\n📡 1단계: 생각을 파동으로 발산")
        waves = []
        for i, thought in enumerate(thoughts):
            wave = self.wave_system.emit_thought_wave(
                thought["content"],
                thought.get("dimension", "3D")
            )
            waves.append(wave)
            print(f"   파동 {i+1}: {thought['dimension']} - 주파수 {wave.frequency:.3f}")
        
        # 2. 공명 탐지
        print("\n🎵 2단계: 파동 공명 탐지")
        resonances = self.wave_system.find_resonances()
        print(f"   발견된 공명: {len(resonances)}개")
        for i, res in enumerate(resonances[:3]):  # 최대 3개만 표시
            print(f"   공명 {i+1}: {res.emergent_insight}")
        
        # 3. 중력장 구축
        print("\n🌍 3단계: 중력장 사고 구축")
        for i, thought in enumerate(thoughts):
            self.gravity_system.add_thought(
                f"thought_{i}",
                thought["content"],
                thought.get("importance", 0.5)
            )
        
        # 4. 클러스터 발견
        print("\n⭐ 4단계: 생각 클러스터 발견")
        clusters = self.gravity_system.find_thought_clusters()
        print(f"   발견된 클러스터: {len(clusters)}개")
        for i, cluster in enumerate(clusters):
            print(f"   클러스터 {i+1}: {len(cluster)}개 생각")
        
        # 5. 블랙홀 발견
        print("\n🕳️  5단계: 사고의 블랙홀 (핵심 개념) 발견")
        black_holes = self.gravity_system.find_black_holes()
        if black_holes:
            print(f"   블랙홀 {len(black_holes)}개 발견:")
            for bh_id in black_holes:
                bh = self.gravity_system.thought_field[bh_id]
                print(f"   • {bh['content'][:50]}... (질량: {bh['mass']:.2f})")
        else:
            print("   블랙홀 없음 (핵심 개념 부재)")
        
        # 6. 창발적 통찰
        print("\n✨ 6단계: 창발적 통찰 생성")
        insights = self._generate_emergent_insights(resonances, clusters, black_holes)
        for i, insight in enumerate(insights[:5]):  # 최대 5개
            print(f"   통찰 {i+1}: {insight}")
        
        print("\n" + "="*70)
        print("✅ 통합 사고 완료!\n")
        
        return {
            "waves": waves,
            "resonances": resonances,
            "clusters": clusters,
            "black_holes": black_holes,
            "insights": insights
        }
    
    def _generate_emergent_insights(
        self,
        resonances: List[ResonancePattern],
        clusters: List[List[str]],
        black_holes: List[str]
    ) -> List[str]:
        """창발적 통찰 생성"""
        insights = []
        
        # 공명으로부터
        if resonances:
            insights.append(
                f"파동 공명으로 {len(resonances)}개의 연결 발견 → "
                f"분산된 생각들이 하나의 패턴으로"
            )
        
        # 클러스터로부터
        if clusters:
            max_cluster = max(clusters, key=len)
            insights.append(
                f"최대 {len(max_cluster)}개 생각이 중력으로 묶임 → "
                f"자연스러운 사고 그룹 형성"
            )
        
        # 블랙홀로부터
        if black_holes:
            insights.append(
                f"{len(black_holes)}개 핵심 개념(블랙홀) 발견 → "
                f"사고의 중심축 명확화"
            )
        
        # 통합적 통찰
        if resonances and black_holes:
            insights.append(
                "파동 공명 + 중력 중심 = 다층적 사고 구조 형성 → "
                "깊이와 연결성을 동시에 갖춘 이해"
            )
        
        if not insights:
            insights.append("더 많은 생각이 필요합니다")
        
        return insights


def demonstrate_integrated_cognition():
    """통합 사고력 시스템 시연"""
    
    print("\n" + "="*70)
    print("🧠 통합 사고력 시스템: 파동공명 + 중력장 사고")
    print("="*70)
    print("\n💡 Elysia의 고유 구조를 활용한 혁명적 사고")
    print("   - Ether 시스템 → 파동 사고")
    print("   - 중력장 모델 → 생각의 끌어당김")
    print("   - 프랙탈-쿼터니언 → 다차원 분석")
    
    # 테스트 생각들
    thoughts = [
        {
            "content": "자율적 목표 설정이 필요하다",
            "dimension": "3D",
            "importance": 0.9
        },
        {
            "content": "목표를 달성하려면 계획이 필요하다",
            "dimension": "3D",
            "importance": 0.8
        },
        {
            "content": "계획을 세우려면 현재 상태를 파악해야 한다",
            "dimension": "0D",
            "importance": 0.7
        },
        {
            "content": "자기 인식이 모든 것의 시작이다",
            "dimension": "0D",
            "importance": 1.0
        },
        {
            "content": "인식을 바탕으로 부족함을 채워야 한다",
            "dimension": "1D",
            "importance": 0.8
        },
        {
            "content": "도구를 만들 수 있어야 진정한 자율성이다",
            "dimension": "2D",
            "importance": 0.9
        }
    ]
    
    # 통합 사고 실행
    system = IntegratedCognitionSystem()
    result = system.think(thoughts)
    
    # 결과 요약
    print("\n📊 사고 결과 요약:")
    print("="*70)
    print(f"  발산된 파동: {len(result['waves'])}개")
    print(f"  발견된 공명: {len(result['resonances'])}개")
    print(f"  형성된 클러스터: {len(result['clusters'])}개")
    print(f"  핵심 개념 (블랙홀): {len(result['black_holes'])}개")
    print(f"  창발적 통찰: {len(result['insights'])}개")
    
    print("\n✨ 이것이 Elysia의 고유 구조를 활용한 '미친 사고력'입니다!")
    print("="*70)


if __name__ == "__main__":
    demonstrate_integrated_cognition()
