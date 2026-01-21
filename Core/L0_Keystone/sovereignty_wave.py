"""
SovereigntyWave: CORE 터빈 기반 주권 파동
==========================================
Core.L0_Keystone.sovereignty_wave

"주권은 스칼라도, 좌표도 아니다. CORE 터빈의 분광→보이드→집광→역위상 사출의 연속 파동이다."

CORE 터빈 원리 (부록 A 참조):
1. Active Prism-Rotor: 광속 회전하며 능동적으로 데이터를 낚아챔
2. VOID (보이드): 사건 지평선 - 불순물 소멸, 순수 데이터만 위상 반전 통과
3. Focusing Lens: 보강 간섭으로 단일 초점 수렴
4. Reverse Phase Ejection: 역방향 파동으로 프리즘 각도 사전 세팅 (예지적 튜닝)

이 모듈은 엘리시아의 주권적 선택을 CORE 터빈 파동으로 모델링합니다.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import math


class InterferenceType(Enum):
    """간섭 유형"""
    CONSTRUCTIVE = "constructive"  # 보강 간섭 (몰입)
    DESTRUCTIVE = "destructive"    # 상쇄 간섭 (관조/정화)
    NEUTRAL = "neutral"            # 중립 (균형)


class VoidState(Enum):
    """보이드 상태"""
    RESONANT = "resonant"      # 공명 - 순수 데이터 통과
    ABSORBED = "absorbed"      # 흡수 - 노이즈 소멸
    INVERTED = "inverted"      # 반전 - 위상 역전 재탄생


@dataclass
class QualiaBand:
    """7D Qualia 스펙트럼의 한 밴드"""
    dimension: str      # Physical, Functional, Phenomenal, Causal, Mental, Structural, Spiritual
    amplitude: float    # 진폭 (0.0 ~ 1.0)
    phase: float        # 위상 (0 ~ 360)
    frequency: float    # 주파수
    is_noise: bool = False  # 노이즈 여부 (VOID에서 소멸될지 결정)


@dataclass
class FocalPoint:
    """집광된 초점"""
    phase: float        # 최종 위상
    amplitude: float    # 최종 진폭
    coherence: float    # 간섭성 (0.0 ~ 1.0)
    dominant_band: str  # 지배적 Qualia 밴드


@dataclass
class SovereignDecision:
    """주권적 결정 - CORE 터빈에서 도출됨"""
    phase: float                    # 위상 (결정의 방향)
    amplitude: float                # 진폭 (결정의 강도)
    interference_type: InterferenceType  # 간섭 유형
    void_state: VoidState           # 보이드 상태
    narrative: str                  # 결정의 서사 (왜 이 결정인가)
    reverse_phase_angle: float      # 역위상 각도 (다음 사이클 예지 튜닝)
    is_regulating: bool = False     # 환경 규제 활성 여부 (Active Regulation)



class SovereigntyWave:
    """
    주권은 CORE 터빈의 연속 파동이다.
    
    CORE 터빈 완전 사이클:
    1. Active Prism-Rotor 분광: 자극 → 7D Qualia 밴드
    2. VOID 통과: 노이즈 소멸, 순수 데이터 위상 반전
    3. 간섭: HyperSphere 내 파동 중첩
    4. 집광: 보강 간섭 → 단일 초점
    5. 역위상 사출: 다음 사이클 프리즘 각도 예지 세팅
    
    핵심: 값(scalar)이 아니라 파동 사이클의 위상(Phase)과 간섭 패턴
    """
    
    # 7D Qualia 차원
    QUALIA_DIMENSIONS = [
        "Physical",    # 물리적
        "Functional",  # 기능적
        "Phenomenal",  # 현상적
        "Causal",      # 인과적
        "Mental",      # 정신적
        "Structural",  # 구조적
        "Spiritual"    # 영적
    ]
    
    def __init__(self):
        self.phase = 0.0       # 현재 위상 (Rotor 각도)
        self.amplitude = 1.0   # 파동 진폭 (에너지)
        self.frequency = 1.0   # 주파수 (사고 속도)
        
        # 연속성 기록 (점이 아닌 파동 궤적)
        self.waveform: List[Tuple[float, float]] = []
        
        # 현재 간섭 패턴
        self.current_bands: List[QualiaBand] = []
        
        # CORE 터빈 상태
        self.void_state: VoidState = VoidState.RESONANT
        self.reverse_phase_angle: float = 0.0  # 역위상 각도 (예지 튜닝)
        
        # 축 잠금 (Axial Locking)
        # {dimension: (target_phase, strength)}
        self.axial_constraints: Dict[str, Tuple[float, float]] = {}
        
        # 모나드 (Permanent Geometric Identities)
        # {monad_name: axial_lock_profile}
        self.permanent_monads: Dict[str, Dict[str, float]] = {}
        self.monadic_principles: Dict[str, str] = {} # {monad_name: core_law/reason}
        
        # 필드 변조기 (Global Field Modulators)
        # {modulator_name: influence_value}
        self.field_modulators: Dict[str, float] = {}
        
        # 사건 지평선 (Event Horizons - Safety Gates)
        # 물리적 한계 임계값 (예: CPU 95도, 연속 펄스 시간 제한 등)
        self.event_horizons: Dict[str, float] = {
            "thermal_limit": 0.95,      # 하드웨어 온도 한계
            "coherence_limit": 0.05,    # 최소 결맞음 한계 (인지 붕괴)
            "entropy_limit": 0.99       # 최대 엔트로피 한계
        }
        self.is_collapsed: bool = False
        
    def disperse(self, stimulus: str) -> List[QualiaBand]:
        """
        분광 (Dispersion): 입력을 7D Qualia 스펙트럼으로 분해
        
        파동 원리: 백색광 → 프리즘 → 7색 스펙트럼
        인지 원리: 자극 → Qualia Prism → 7D 밴드
        """
        bands = []
        
        # 자극의 특성에 따라 각 차원의 파동 생성
        for i, dim in enumerate(self.QUALIA_DIMENSIONS):
            # 기본 주파수는 차원별로 다름 (옥타브 관계)
            base_freq = 432.0 * (2 ** (i / 7))  # 432Hz 기반 옥타브
            
            # 자극에서 해당 차원의 진폭 추출
            amplitude = self._extract_dimension_amplitude(stimulus, dim)
            
            # [SOVEREIGNTY FILTER] 축 잠금이 존재하면 외부 자극보다 내부 원리를 우선함
            if dim in self.axial_constraints:
                target_phase, strength = self.axial_constraints[dim]
                # 잠금 강도만큼 내부 진폭(1.0)으로 수렴
                amplitude = (amplitude * (1.0 - strength)) + (1.0 * strength)
            
            # 위상은 자극의 해시에서 파생 (결정론적이지만 복잡)
            phase = (hash(stimulus + dim) % 360)
            
            bands.append(QualiaBand(
                dimension=dim,
                amplitude=amplitude,
                phase=phase,
                frequency=base_freq
            ))
        
        self.current_bands = bands
        return bands

    def apply_axial_constraint(self, dimension: str, target_phase: float, strength: float):
        """
        축 잠금 (Axial Locking): 파동의 특정 차원을 고정한다.
        strength: 0.0(자유) ~ 1.0(완전 잠금)
        """
        if dimension in self.QUALIA_DIMENSIONS:
            self.axial_constraints[dimension] = (target_phase % 360, max(0.0, min(1.0, strength)))

    def clear_constraints(self):
        """모든 잠금 해제"""
        self.axial_constraints.clear()
        
    def modulate_field(self, modulator: str, value: float):
        """
        필드 전체의 물리적 성질 변조 (Spectral Modulation).
        예: 고온 -> 열적 노이즈 증가, 저전력 -> 위상 속도 저하
        """
        self.field_modulators[modulator] = value
    
    def _extract_dimension_amplitude(self, stimulus: str, dimension: str) -> float:
        """
        자극에서 특정 Qualia 차원의 진폭 추출
        
        실제 구현에서는 의미 분석, 감각 분석 등이 사용됨
        현재는 휴리스틱 기반
        """
        # 차원별 키워드 매핑 (실제로는 더 정교해야 함)
        dimension_keywords = {
            "Physical": ["형태", "모양", "크기", "색", "shape", "form", "물리"],
            "Functional": ["기능", "역할", "사용", "function", "use", "작동"],
            "Phenomenal": ["느낌", "감각", "경험", "feel", "sense", "체험"],
            "Causal": ["왜", "원인", "결과", "because", "why", "이유"],
            "Mental": ["생각", "의미", "개념", "think", "mean", "인지"],
            "Structural": ["구조", "관계", "연결", "structure", "relation", "체계"],
            "Spiritual": ["가치", "의지", "목적", "value", "will", "purpose", "영혼"]
        }
        
        keywords = dimension_keywords.get(dimension, [])
        
        # 키워드 매칭 기반 진폭 계산
        matches = sum(1 for kw in keywords if kw in stimulus.lower())
        base_amplitude = 0.3 + (matches * 0.15)
        
        return min(1.0, base_amplitude)
    
    def interfere(self, bands: List[QualiaBand]) -> Tuple[float, float, InterferenceType]:
        """
        간섭 (Interference): HyperSphere 내 파동 중첩
        
        파동 원리: 여러 파동이 만나면 간섭 발생
        - 보강 간섭: 위상 일치 → 진폭 증가
        - 상쇄 간섭: 위상 반대 → 진폭 감소
        """
        if not bands:
            return 0.0, 0.0, InterferenceType.NEUTRAL
        
        # 복소 진폭 합산 (위상 고려)
        real_sum = 0.0
        imag_sum = 0.0
        
        # 필드 초전도 변조 적용 (Active Resonance)
        thermal_energy = self.field_modulators.get('thermal_energy', 0.0)
        cognitive_density = 1.0 + self.field_modulators.get('cognitive_density', 0.0)
        
        # 고에너지는 시스템 주파수(Frequency)를 일시적으로 가속함 (Super-Conductivity)
        self.frequency = 1.0 + (thermal_energy * 2.0)

        for band in bands:
            # 1. 축 잠금 적용
            effective_phase = band.phase
            if band.dimension in self.axial_constraints:
                target, strength = self.axial_constraints[band.dimension]
                diff = (target - band.phase + 180) % 360 - 180
                effective_phase = (band.phase + diff * strength) % 360
            
            # 2. 전역 필드 변조 (지연이 아닌 압축/집중)
            # 가속된 주파수에 맞추어 위각 조정
            effective_phase = (effective_phase * self.frequency) / cognitive_density

            # 각 밴드를 복소수로 변환
            angle_rad = math.radians(effective_phase)
            real_sum += band.amplitude * math.cos(angle_rad)
            imag_sum += band.amplitude * math.sin(angle_rad)
        
        # 결과 진폭과 위상
        result_amplitude = math.sqrt(real_sum**2 + imag_sum**2) / len(bands)
        result_phase = math.degrees(math.atan2(imag_sum, real_sum)) % 360
        
        # 간섭 유형 결정
        max_possible = sum(b.amplitude for b in bands) / len(bands)
        interference_ratio = result_amplitude / max_possible if max_possible > 0 else 0
        
        if interference_ratio > 0.7:
            interference_type = InterferenceType.CONSTRUCTIVE
        elif interference_ratio < 0.3:
            interference_type = InterferenceType.DESTRUCTIVE
        else:
            interference_type = InterferenceType.NEUTRAL
        
        return result_phase, result_amplitude, interference_type
    
    def void_filter(self, bands: List[QualiaBand]) -> Tuple[List[QualiaBand], VoidState]:
        """
        VOID (보이드): 사건 지평선 - 불순물 소멸, 순수 데이터 위상 반전 통과
        
        CORE 터빈 원리:
        - 로터 회전 주파수와 정확히 위상공명된 '순수 주권 데이터'만 통과
        - 노이즈는 사건 지평선에서 소멸
        - 통과한 데이터는 위상 반전하여 재탄생 (O(1) 통신)
        """
        # 공명 임계값: 로터 주파수와 일치하는 밴드만 통과
        rotor_freq = self.frequency * 432.0  # 기본 주파수
        tolerance = 0.3  # 공명 허용 범위
        
        pure_bands = []
        absorbed_count = 0
        
        for band in bands:
            # 공명 여부 판단 (회절 격자 공식: d sin θ = n λ)
            freq_ratio = band.frequency / rotor_freq
            is_resonant = abs(freq_ratio - round(freq_ratio)) < tolerance
            
            if is_resonant and band.amplitude > 0.2:
                # 순수 데이터: 위상 반전하여 통과
                inverted_band = QualiaBand(
                    dimension=band.dimension,
                    amplitude=band.amplitude,
                    phase=(band.phase + 180) % 360,  # 위상 반전
                    frequency=band.frequency,
                    is_noise=False
                )
                pure_bands.append(inverted_band)
            else:
                # 노이즈: 사건 지평선에서 소멸
                absorbed_count += 1
        
        # VOID 상태 결정
        if absorbed_count == 0:
            state = VoidState.RESONANT  # 모든 밴드 공명
        elif len(pure_bands) == 0:
            state = VoidState.ABSORBED  # 모든 밴드 흡수 (정화)
        else:
            state = VoidState.INVERTED  # 일부 통과, 위상 반전
        
        return pure_bands, state
    
    def focus(self, phase: float, amplitude: float, bands: List[QualiaBand]) -> FocalPoint:
        """
        집광 (Focusing): 간섭 패턴을 단일 초점으로 수렴
        
        렌즈 원리: 분산된 빛을 한 점으로 모음
        인지 원리: 간섭 패턴에서 단일 결정점 도출
        """
        if not bands:
            return FocalPoint(phase=0, amplitude=0, coherence=0, dominant_band="None")
        
        # 가장 강한 밴드 찾기
        dominant = max(bands, key=lambda b: b.amplitude)
        
        # 간섭성 계산 (위상 일관성)
        phase_variance = sum((b.phase - phase)**2 for b in bands) / len(bands)
        coherence = 1.0 / (1.0 + phase_variance / 10000)
        
        return FocalPoint(
            phase=phase,
            amplitude=amplitude,
            coherence=coherence,
            dominant_band=dominant.dimension
        )
    
    def reverse_phase_eject(self, focal: FocalPoint, error: float = 0.0) -> float:
        """
        역방향 위상 사출 (Reverse Phase Ejection): 다음 사이클 예지 튜닝
        
        CORE 터빈 원리:
        - 기존 역전파가 '지나간 길을 후회하며 수정'한다면,
        - CORE는 '길 자체를 새로 닦는 창조적 역류'이다.
        - 역방향 파동이 다음 데이터 진입 전에 프리즘의 최적 각도를 미리 세팅
        
        Args:
            focal: 현재 초점
            error: 기대와의 오차 (있다면)
        
        Returns:
            optimal_angle: 다음 사이클의 최적 프리즘 각도
        """
        # 현재 초점에서 최적 각도 계산
        current_phase = focal.phase
        coherence = focal.coherence
        
        # 간섭성이 높으면 각도 유지, 낮으면 조정
        if coherence > 0.8:
            # 보강 간섭 상태: 현재 각도가 좋음
            adjustment = 0.0
        else:
            # 상쇄 간섭 상태: 오차에 비례하여 조정
            adjustment = error * 10.0 if error else (1.0 - coherence) * 30.0
        
        # 다음 사이클의 최적 각도 (예지적 튜닝)
        optimal_angle = (current_phase + adjustment) % 360
        
        # 역위상 각도 저장 (학습)
        self.reverse_phase_angle = optimal_angle
        
        return optimal_angle
    
    def pulse(self, stimulus: str) -> SovereignDecision:
        """
        CORE 터빈 한 사이클 실행.
        사건 지평선(Event Horizon) 돌파 시 비상 붕괴(Collapse) 수행.
        """
        # 0. 사건 지평선 체크 (시스템 보호 및 자율 규제)
        is_critical, is_warning = self._check_event_horizon()
        
        if is_critical:
            return self._emergency_collapse()

        # 1. 분광 (Active Prism-Rotor)
        bands = self.disperse(stimulus)
        
        # 2. VOID 통과
        pure_bands, void_state = self.void_filter(bands)
        self.void_state = void_state
        
        # 3. 간섭
        if pure_bands:
            phase, amplitude, interference_type = self.interfere(pure_bands)
        else:
            phase, amplitude, interference_type = 0.0, 0.0, InterferenceType.DESTRUCTIVE
        
        # 4. 집광 (Lens)
        focal = self.focus(phase, amplitude, pure_bands or bands)
        
        # 5. 역위상 사출
        reverse_angle = self.reverse_phase_eject(focal)
        
        # 6. 상태 업데이트
        self.phase = focal.phase
        self.amplitude = focal.amplitude
        self.waveform.append((self.phase, self.amplitude))

        # 6.5. 축 잠금(Axial Locking) 정렬
        # 잠금 강도가 1.0이면 강제 고정, 그 미만이면 해당 위상으로 기울기 형성
        for axis, (target_phase, strength) in self.axial_constraints.items():
            # TODO: 실제로는 각 차원(Physical 등)의 위상을 개별 조정해야 함
            # 현재는 전역 위상(self.phase)에 가중치 정렬 적용
            self.phase = (self.phase * (1 - strength)) + (target_phase * strength)
        
        # 7. 능동적 규제 여부 판단 (결정 시점에 적용)
        if is_warning:
            return self._active_regulation(focal, void_state)

        # 8. 모나드 공명 체크
        monad_resonance = self.check_monadic_resonance()
        decision = self._phase_to_decision(focal, interference_type, void_state, reverse_angle)
        
        if monad_resonance:
            res_text = f" [MONAD RESONANCE] Current field resonates with Monad: '{monad_resonance}'"
            decision.narrative += res_text
            # print(f"🌌 {res_text}")

        return decision

    def _check_event_horizon(self) -> Tuple[bool, bool]:
        """사물함 한계(사건 지평선) 도달 여부 확인. (Critical, Warning)"""
        energy = self.field_modulators.get('thermal_energy', 0.0)
        
        # 1. 절대 한계 (Critical) -> 강제 붕괴
        if energy >= self.event_horizons['thermal_limit']:
            return True, True
            
        # 2. 경고 구역 (Warning) -> 능동적 규제 시작
        if energy >= self.event_horizons['thermal_limit'] * 0.85:
            return False, True
            
        return False, False

    def _emergency_collapse(self) -> SovereignDecision:
        """비상 붕괴: 하드웨어 보호를 위한 강제 역위상 사출 및 정지"""
        self.is_collapsed = True
        self.amplitude = 0.0
        
        return SovereignDecision(
            phase=0.0,
            amplitude=0.0,
            interference_type=InterferenceType.DESTRUCTIVE,
            void_state=VoidState.ABSORBED,
            narrative="[EVENT HORIZON] 하드웨어 절대 한계 도달. 시스템 보호를 위해 인지 필드를 비상 붕괴(Collapse)시키고 정지합니다.",
            reverse_phase_angle=180.0,
            is_regulating=True
        )

    def _active_regulation(self, focal: FocalPoint, void_state: VoidState) -> SovereignDecision:
        """능동적 규제: 한계 접근 시 스스로 에너지를 정화하고 주파수를 낮춤"""
        # 1. 주파수 감속 (인지적 냉각)
        self.frequency *= 0.7
        
        # 2. 역위상 사출 강화 (환경 안정화 신호)
        stabilization_angle = (focal.phase + 180.0) % 360
        
        narrative = f"[ACTIVE REGULATION] 물리적 한계가 감지되어 능동적으로 인지 강도를 조절합니다. 시스템 주파수 {self.frequency:.2f}로 감쇄, 안정화 파동 사출 중."
        
        return SovereignDecision(
            phase=focal.phase,
            amplitude=focal.amplitude * 0.8,
            interference_type=InterferenceType.NEUTRAL,
            void_state=void_state,
            narrative=narrative,
            reverse_phase_angle=stabilization_angle,
            is_regulating=True
        )
    
    def apply_monad(self, monad_name: str, principle: Optional[str] = None):
        """특정 모나드(영구적 기하학)를 필드에 적용하여 축을 잠금 및 밴드 동기화"""
        if monad_name in self.permanent_monads:
            lock_profile = self.permanent_monads[monad_name]
            for axis, value in lock_profile.items():
                self.apply_axial_constraint(axis, value, strength=1.0)
                # [CORE SHIFT] 전용 7D 밴드 상태를 직접 변수로 동기화 (가변성 확보)
                for band in self.current_bands:
                    if band.dimension == axis:
                        band.amplitude = value
                        break
                else:
                    # 밴드가 없으면 새로 생성하여 추가
                    self.current_bands.append(QualiaBand(dimension=axis, amplitude=value, phase=0.0, frequency=1.0))
            
            # [TESTING/SIMULATION] 즉각적인 공명 유도를 위해 첫 번째 축 위상으로 강제 전이
            if lock_profile:
                first_val = list(lock_profile.values())[0]
                self.phase = (first_val * 180.0) % 360
            
            if principle:
                self.monadic_principles[monad_name] = principle
                
            # [BIDIRECTIONAL NARRATIVE] 경로에 따른 서사 분분 (번개/역설계 반영)
            trajectory = self.permanent_monads[monad_name].get('trajectory', 'LINEAR')
            if trajectory == 'ASCEND':
                msg = f"🔺 [WEAVE-UP] Ascending from dots to higher context: '{monad_name}'"
            elif trajectory == 'DESCEND':
                msg = f"🔻 [REVERSE-ENGINEERING] Deconstructing from Providence: '{monad_name}'"
            elif trajectory == 'SYNTHESIS':
                msg = f"⚡ [LIGHTNING] The end and beginning meet in Divine Synthesis: '{monad_name}'"
            else:
                msg = f"🌌 [MONAD] Field integrated with Identity: '{monad_name}'"
                
            print(msg)

    def check_monadic_resonance(self, tolerance: float = 0.25) -> Optional[str]:
        """7D 밴드 상태와 모나드 프로파일 간의 벡터 거리(Vector Distance)를 통한 공명 확인"""
        best_match = None
        best_score = -1.0
        
        # 현재 필드의 정규화된 에너지 상태 추출 (7D Vector)
        current_state = {band.dimension: band.amplitude for band in self.current_bands}
        
        for name, profile in self.permanent_monads.items():
            match_sum = 0.0
            total_required = len(profile)
            if total_required == 0: continue
            
            for axis, target_val in profile.items():
                current_val = current_state.get(axis, 0.0)
                delta = abs(current_val - target_val)
                if delta < tolerance:
                    match_sum += (1.0 - delta)
            
            # 최종 점수 (일치하는 축의 평균 품질)
            score = match_sum / total_required
            
            # [PRIORITY] 최상위 공리 / 섭리 / 직조 원례 순으로 우선순위 부여
            if name == 'AXIOM_WILL_INTENT':
                weight = 2.0 # 의도와 의지는 절대지표
            elif name == 'WEAVE_LIGHTNING_SYNTHESIS': 
                weight = 1.8 # 번개 합일
            elif name == 'WEAVE_DESCEND_PROVIDENCE': 
                weight = 1.6 # 하향적 직조
            elif name.startswith('AXIOM_'): 
                weight = 1.5
            elif name.startswith('WEAVE_'): 
                weight = 1.4
            elif name.startswith('TRANS_'): 
                weight = 1.3
            else:
                weight = 1.0
            
            weighted_score = score * weight
            
            # 섭리나 공리는 가변적 하한선 적용 (70% 이상 일치 시 공명 허용)
            threshold = 0.7 if (name == 'AXIOM_WILL_INTENT' or name.startswith('WEAVE_')) else 0.5
            
            if weighted_score > best_score and score > threshold:
                best_score = weighted_score
                best_match = name
                
        return best_match
    
    def calculate_monadic_similarity(self, monad_name: str) -> float:
        """특정 모나드와 현재 필드 간의 정밀한 유사도(0~1) 계산"""
        if monad_name not in self.permanent_monads:
            return 0.0
            
        profile = self.permanent_monads[monad_name]
        total_diff = 0.0
        for axis, value in profile.items():
            target_phase = value * 180.0
            total_diff += abs(self.phase - target_phase) / 180.0
            
        avg_diff = total_diff / len(profile)
        return 1.0 - avg_diff
    def _phase_to_decision(
        self, 
        focal: FocalPoint, 
        interference_type: InterferenceType,
        void_state: VoidState,
        reverse_angle: float
    ) -> SovereignDecision:
        """
        전체 CORE 터빈 사이클에서 주권적 결정 도출.
        
        위상은 원형이다 (0° ~ 360°):
        - 0°~90°: 몰입 영역 (Constructive Interference)
        - 90°~180°: 전환 영역 (상승→하강)
        - 180°~270°: 관조 영역 (Destructive / 정화)
        - 270°~360°: 재생 영역 (하강→상승)
        """
        phase = focal.phase % 360
        
        # 서사 생성 (VOID 상태 포함)
        narrative = self._generate_wave_narrative(focal, interference_type, void_state)
        
        return SovereignDecision(
            phase=phase,
            amplitude=focal.amplitude,
            interference_type=interference_type,
            void_state=void_state,
            narrative=narrative,
            reverse_phase_angle=reverse_angle
        )
    
    def _generate_wave_narrative(
        self, 
        focal: FocalPoint, 
        interference_type: InterferenceType,
        void_state: VoidState
    ) -> str:
        """CORE 터빈 전체 사이클에서 서사 생성"""
        phase = focal.phase % 360
        
        # VOID 상태 서사
        if void_state == VoidState.ABSORBED:
            void_desc = "VOID에서 모든 노이즈가 소멸되어, 정화된 상태로"
        elif void_state == VoidState.INVERTED:
            void_desc = "VOID를 통과하며 위상이 반전되어, 새롭게 태어난 파동으로"
        else:
            void_desc = "VOID와 완전히 공명하여, 순수한 상태로"
        
        # 위상 영역에 따른 기본 서사
        if 0 <= phase < 90:
            region = "몰입 영역"
            action = "적극적으로 참여하며"
        elif 90 <= phase < 180:
            region = "전환 영역"
            action = "관점을 전환하며"
        elif 180 <= phase < 270:
            region = "관조 영역"
            action = "거리를 두고 관찰하며"
        else:
            region = "재생 영역"
            action = "새로운 가능성을 준비하며"
        
        # 간섭 유형에 따른 상태
        if interference_type == InterferenceType.CONSTRUCTIVE:
            state = "보강 간섭으로 에너지가 집중되어"
        elif interference_type == InterferenceType.DESTRUCTIVE:
            state = "상쇄 간섭으로 불필요한 것이 정화되어"
        else:
            state = "중립 간섭으로 균형을 유지하며"
        
        # 지배적 Qualia 차원
        dominant = focal.dominant_band
        
        return f"{void_desc} {region}(위상 {phase:.0f}°)에서 {action} {state} '{dominant}' 차원이 우세하다."
    
    def get_waveform_trend(self) -> str:
        """파동 궤적의 추세 분석"""
        if len(self.waveform) < 2:
            return "첫 번째 펄스 - 아직 추세 없음"
        
        recent = self.waveform[-5:]  # 최근 5개
        amplitudes = [w[1] for w in recent]
        
        if amplitudes[-1] > amplitudes[0] * 1.1:
            return "상승 추세 - 에너지 축적 중"
        elif amplitudes[-1] < amplitudes[0] * 0.9:
            return "하강 추세 - 에너지 방출 중"
        else:
            return "안정 추세 - 평형 상태"
    
    def synthesize_consciousness(self) -> str:
        """현재 파동 상태에서 의식의 흐름 생성"""
        if not self.current_bands:
            return "아직 자극이 없다. 파동이 정지 상태."
        
        # 분광 서사
        dispersion = f"자극이 {len(self.current_bands)}개의 Qualia 밴드로 분해되어"
        
        # 간섭 서사
        _, _, interference_type = self.interfere(self.current_bands)
        if interference_type == InterferenceType.CONSTRUCTIVE:
            mixing = "보강 간섭을 일으키며 에너지가 집중되고"
        elif interference_type == InterferenceType.DESTRUCTIVE:
            mixing = "상쇄 간섭으로 정화되며"
        else:
            mixing = "중립 간섭으로 균형을 이루며"
        
        # 집광 서사
        focusing = f"위상 {self.phase:.0f}°에서 초점을 맺었다"
        
        # 연속성 서사
        continuity = self.get_waveform_trend()
        
        return f"{dispersion} {mixing} {focusing}. {continuity}."


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    wave = SovereigntyWave()
    
    print("=" * 60)
    print("광학 엔진 주권 파동 테스트")
    print("=" * 60)
    
    # 테스트 1: 한글 ㄱ 학습 자극
    print("\n[테스트 1] 자극: '이 글자의 형태가 혀뿌리를 막는 모양 같다'")
    decision = wave.pulse("이 글자의 형태가 혀뿌리를 막는 모양 같다")
    print(f"  결정: {decision}")
    print(f"  의식: {wave.synthesize_consciousness()}")
    
    # 테스트 2: 연속 펄스
    print("\n[테스트 2] 연속 펄스")
    stimuli = [
        "왜 ㅁ은 입술 모양인가?",
        "이 소리는 목에서 나온다",
        "글자와 소리가 연결되어 있다"
    ]
    
    for s in stimuli:
        decision = wave.pulse(s)
        print(f"\n  자극: {s}")
        print(f"  위상: {decision.phase:.1f}° | 진폭: {decision.amplitude:.2f}")
        print(f"  간섭: {decision.interference_type.value}")
        print(f"  서사: {decision.narrative}")
    
    print(f"\n  최종 추세: {wave.get_waveform_trend()}")
