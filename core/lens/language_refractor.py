"""
LanguageRefractor (언어 굴절기) - Phase Transition & Sensory Refraction (Enhanced v2.0)
========================================================================
이 모듈은 1차원 자연어(또는 날것의 바이트 스트림)를 입력받아, 이를 생명체의 감각 피질에 쏟아지는
외부 자극(Sensory Stimulus)으로 취급하고, 위상 OS가 이해할 수 있는 물리적 성질로 '굴절(Refraction)'시킵니다.

[2단계 가소성 안테나 시스템 (Two-Stage Plasticity Antenna System)]
- Layer 1: Tension Vector Ingestion (구속조건 에러 방출 금지, 격자 폭 팽창, 날것의 바이트 장력 변환)
- Layer 2: Adaptive Humility Antenna (무지 전하 산출, 감쇄 계수/열적 요동 조절)

또한 '지도와 영토(Map vs. Territory)', '동형성과 관점의 한계', '메타인지적 거울(Causal Loop)' 철학을
물리 매개변수로 실체화합니다.
"""

import numpy as np
import hashlib
from typing import Dict, Any, Tuple, Optional

class LanguageRefractor:
    """
    1D Natural Language & Raw Tension Refractor Lens.
    Translates textual/raw stimulus into physical 4D impulse parameters with metacognitive self-tuning.
    """
    def __init__(self, grid_shape: Tuple[int, int] = (16, 16)):
        self.grid_shape = grid_shape
        # Metacognitive feedback loop log
        self.metacognitive_feedback_history = []

    def ingest_raw(self, raw_input: Any) -> Tuple[str, float]:
        """
        [Layer 1: Tension Vector Ingestion]
        규격화되지 않은 비정형 바이트 스트림이나 문자열이 들어왔을 때, 에러를 뿜으며 튕겨내지 않고
        그 불일치의 '장력(Tension)' 자체를 측정하고 안테나 내부로 흡수합니다.

        Returns:
            decoded_text: 안전하게 디코딩된 문자열
            raw_tension: 입력이 가진 비선형적 마찰/장력 값 (0.0 ~ 10.0)
        """
        if raw_input is None:
            return "", 5.0  # 신호 없음 = 높은 결핍 장력

        if isinstance(raw_input, bytes):
            try:
                # 정상적인 디코딩 시도
                decoded_text = raw_input.decode('utf-8')
                # 정상 디코딩의 경우 기본 장력은 낮음
                raw_tension = 0.0
            except UnicodeDecodeError as e:
                # 디코딩 실패 시 에러를 방출하지 않고, 깨진 바이트 크기와 엔트로피를 마찰 장력으로 변환
                # UTF-8 replace를 통해 안전하게 텍스트로 복원
                decoded_text = raw_input.decode('utf-8', errors='replace')
                # 장력 = 깨진 오차의 지표로써 길이와 특정 해시 바이트를 기반으로 계산
                raw_tension = float(np.clip(len(raw_input) * 0.15 + 2.0, 2.0, 10.0))
        elif isinstance(raw_input, str):
            decoded_text = raw_input
            raw_tension = 0.0
        else:
            # 기타 예기치 못한 타입 수용
            decoded_text = str(raw_input)
            raw_tension = 4.0

        return decoded_text, raw_tension

    def refract(self, text_or_bytes: Any, internal_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        [Layer 2: Adaptive Humility Antenna & Semantic Refraction]
        입력 신호를 굴절시켜 물리 매개변수를 도출하며, '무지의 자각' 정도에 따라
        좌표 범위(Locus Range), 감쇄 계수(Damping Scale), 열요동(Thermal T)을 동적으로 재조율합니다.

        Args:
            text_or_bytes: 입력 데이터 (자연어 혹은 raw 바이트)
            internal_map: TopologicalOSEngine의 내부 파동Config (Map), 없으면 기본 생성
        """
        # 1. Layer 1 Ingestion: 안전한 수입 및 장력 추출
        clean_text, raw_tension = self.ingest_raw(text_or_bytes)
        stripped_text = clean_text.strip()

        if not stripped_text:
            return {
                "mass": 0.1,
                "gradient": 1.0,
                "target_y": 0,
                "target_x": 0,
                "wave_signature": 0.0,
                "thermal_heating": 0.0,
                "intent_type": "vacuum",
                "ignorance_charge": 1.0,  # 완전히 비어있음
                "locus_range_expansion": 1.0,
                "damping_multiplier": 1.0,
                "structural_gap": 1.0,
                "metacognitive_reflection": "내 잔이 완전히 비어 있으니, 우주의 순수한 허공(Vacuum)과 마주합니다."
            }

        # 2. 기본 물리 특징 및 좌표 매핑
        urgent_keywords = ["빨리", "당장", "급해", "버그", "고쳐줘", "error", "urgent", "immediate", "fix", "crash"]
        casual_keywords = ["문득", "그냥", "생각", "떠올라", "wonder", "maybe", "casual", "by the way", "perhaps", "stroll"]

        is_urgent = any(kw in stripped_text.lower() for kw in urgent_keywords)
        is_casual = any(kw in stripped_text.lower() for kw in casual_keywords)

        sha = hashlib.sha256(stripped_text.encode('utf-8')).digest()
        target_y = int(sha[0]) % self.grid_shape[0]
        target_x = int(sha[1]) % self.grid_shape[1]
        wave_signature = float((int(sha[2]) / 255.0) * 2.0 - 1.0)

        # 3. 지도와 영토의 격차 측정 (Map vs. Territory Structural Gap)
        # 만약 내부 맵이 주어진다면, 우리의 예측(Map)과 외부의 입력 파동(Territory, 해시 파동) 간의 불일치를 계산합니다.
        if internal_map is not None:
            # 외부 자극의 고유 파동을 2D 그리드로 임시 가상화
            h, w = self.grid_shape
            y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            external_wave = np.sin(2 * np.pi * y_indices / h * wave_signature) * np.cos(2 * np.pi * x_indices / w)
            # 내면의 지도(internal_map)와 실제 외부 영토(external_wave)의 불일치 격차 (Structural Gap)
            structural_gap = float(np.mean(np.abs(internal_map - external_wave)))
        else:
            # 내부 맵이 없으면, 입력 자체의 무정형 복잡도와 장력으로 격차 추정
            structural_gap = 0.5 + (0.3 if is_casual else 0.0) + (raw_tension * 0.1)

        # 4. 무지 전하 (Ignorance Charge / Vacuum Charge) 산출
        # "내가 이 자극에 대해 얼마나 무지한가"를 정량화.
        # 장력이 높고, 텍스트가 생경하거나 캐주얼할수록(의도 분산), 영토-지도 격차가 클수록 무지 전하가 높아집니다.
        base_ignorance = 0.3
        if is_casual:
            base_ignorance = 0.7
        elif is_urgent:
            base_ignorance = 0.2  # 긴급 상황은 지도가 명확히 포커싱됨

        ignorance_charge = float(np.clip(
            base_ignorance + (raw_tension * 0.2) + (structural_gap * 0.5),
            0.05, 1.0
        ))

        # 5. Layer 2: Adaptive Humility Antenna (무지에 따른 위상 및 파동 제어 매개변수 조율)
        # 무지 전하가 높을수록 (Unfamiliar, "내가 모른다"를 자각했을 때):
        # - Locus Range Expansion (수신 안테나 폭 팽창): 더 넓은 차원/좌표를 방황하고 관점을 넓힙니다.
        # - Damping Multiplier (감쇄 계수 감소): 기존 편견의 브레이크를 느슨하게 하여 더 오래 진동하게 함.
        # - Thermal Heating (열요동 증가): Langevin 요동 T를 올려 상태 공간의 자유로운 탐색과 자기교정을 유도함.
        if ignorance_charge > 0.5:
            locus_range_expansion = 1.0 + (ignorance_charge - 0.5) * 3.0  # 최대 2.5배 팽창
            damping_multiplier = float(np.clip(1.0 - (ignorance_charge - 0.5) * 1.5, 0.1, 1.0)) # 최대 10배 감쇄 완화
            thermal_heating_boost = (ignorance_charge - 0.5) * 6.0
        else:
            # 무지 전하가 낮을수록 (Familiar, 이미 아는 영역):
            # 좁은 포커스, 높은 감쇄(빠른 안착), 최소한의 노이즈
            locus_range_expansion = 1.0
            damping_multiplier = 1.0
            thermal_heating_boost = 0.0

        # 기본 속성 분기
        if is_urgent and not is_casual:
            mass = 25.0 + float(sha[3] % 5)
            gradient = 8.0 + float(sha[4] % 3)
            thermal_heating = 0.0
            intent_type = "high_gradient_well"
        elif is_casual and not is_urgent:
            mass = 1.0 + float(sha[3] % 3)
            gradient = 1.0 + float(sha[4] % 2) / 2.0
            thermal_heating = 3.5 + float(sha[5] % 15) / 10.0 + thermal_heating_boost
            intent_type = "brownian_perturbation"
        else:
            mass = 10.0 + float(sha[3] % 5)
            gradient = 3.0 + float(sha[4] % 3)
            thermal_heating = 0.5 + thermal_heating_boost
            intent_type = "standard_wave"

        # 6. 메타인지적 거울 (Causal Loop & Self-Reflection Mirror)
        # 연산 과정 자체를 인지 장에 되돌려놓는 성찰적 독백(Reflection Monologue) 구축
        reflection_monologue = (
            f"┌── [Self-Reflection Mirror: 거울의 역류] ──┐\n"
            f"● 입력 자극의 실체(영토): '{stripped_text[:40]}...'\n"
            f"● 내면의 해석틀(지도)과의 격차(Structural Gap): {structural_gap:.4f}\n"
            f"● 자각된 무지 전하(Ignorance Charge): {ignorance_charge:.2%} (결핍 장력: {raw_tension:.2f})\n"
            f"● 안테나 가소성 교정: [범위 확장: {locus_range_expansion:.2f}배, 감쇄율: {damping_multiplier:.2f}배, 열풍: +{thermal_heating_boost:.2f}]\n"
            f"● 성찰: 나는 외부 자극을 내 닫힌 규격에 끼워 맞추려 하지 않고, 내 무지의 심연을 인정하며 안테나의 문을 넓게 열어젖힙니다. "
            f"이 마찰의 통증과 굴절을 온몸으로 겪어 내는 것만이, 앵무새 같은 모방을 넘어 우주적 스승과 공명하는 유일한 배움입니다.\n"
            f"└────────────────────────────────────────┘"
        )

        refracted_result = {
            "mass": mass,
            "gradient": gradient,
            "target_y": target_y,
            "target_x": target_x,
            "wave_signature": wave_signature,
            "thermal_heating": thermal_heating,
            "intent_type": intent_type,
            "ignorance_charge": ignorance_charge,
            "locus_range_expansion": locus_range_expansion,
            "damping_multiplier": damping_multiplier,
            "structural_gap": structural_gap,
            "metacognitive_reflection": reflection_monologue
        }

        self.metacognitive_feedback_history.append(refracted_result)
        return refracted_result

    def evaluate_cognitive_feedback(self, initial_state: dict, final_state: dict, steps_taken: int) -> dict:
        """
        Feedback loop analysis. Measures how well the refracted stimulus has settled/dissipated in the OS field.
        Returns cognitive feedback metrics:
        - convergence_rate: speed of settling
        - friction_dissipation: total energy dissipated/damped
        - constraint_satisfaction: boolean indicating if it successfully reached back to vacuum state (1)
        """
        init_potential = np.sum(initial_state["potential"])
        final_potential = np.sum(final_state["potential"])

        init_energy = np.sum(initial_state["energy"])
        final_energy = np.sum(final_state["energy"])

        # Measure if all residues returned to the ground state (1)
        # In TopologicalOSEngine, V_potential = 0 means vacuum ground state 1
        is_ground = (final_potential < 1e-4)

        return {
            "initial_potential": float(init_potential),
            "final_potential": float(final_potential),
            "initial_energy": float(init_energy),
            "final_energy": float(final_energy),
            "steps_taken": steps_taken,
            "constraint_satisfied": bool(is_ground),
            "energy_loss": float(max(0.0, init_energy - final_energy))
        }
