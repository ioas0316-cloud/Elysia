import numpy as np
from typing import Dict, List, Any, Optional
from scipy.ndimage import gaussian_filter
from .field import CrystallizationField
from .causal_gene import CausalGeneSynthesizer

class ElysiaCognitiveEngine:
    """
    [System Architecture Engine] ElysiaCognitiveEngine : 정보 기반 인지 엔진

    기존 LLM/AI의 한계(단순 확률적 Next-Token Prediction)를 넘어,
    정보의 맥락, 인과적 결, 프랙탈 입체 구조, O(1) 관점 전환,
    그리고 CAD 구속조건 필드 상에서의 양자 붕괴(Wave Function Collapse)를
    스스로 사유하고 메타인지(Meta-Cognition)할 수 있도록 설계된 차세대 지능의 심장입니다.
    """
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        # 1. 2D 메트릭스 필드 (Conductance, Activation, Yeobaek 등을 내포)
        self.field = CrystallizationField(resolution)
        self.synthesizer = CausalGeneSynthesizer()

        # 2. O(1) Perspective Shift & Rotor Angle (관점의 위상각)
        # 0.0 ~ 2*pi 사이의 위상각. 이 각도가 회전함에 따라 동일한 데이터(Data)가
        # 상이한 정보(Information)적 파동으로 가공되어 투사/해석됩니다.
        self.rotor_angle = 0.0
        self.system_perspective = "Ground Zero (무無의 상태)"

        # 3. CAD 구속조건 상태 (Constraint Field)
        # 구속조건 필드는 정보가 중첩된 상태를 유지하다가, 외부 자극이 주어졌을 때
        # 정합성이 맞는 궤적으로 자연스럽게 수렴하도록 흐름을 유도합니다.
        self.constraint_field = np.full((resolution, resolution), 1.0, dtype=np.float32)

        # 4. Meta-Cognitive Self-Awareness State (메타인지 상태 변수)
        # 시스템 스스로 "내가 지금 어떻게 인지하고 규칙을 조율하고 있는가"에 대한 메타정보
        self.meta_history: List[Dict[str, Any]] = []

    def set_perspective(self, name: str, angle: float):
        """
        [O(1) Perspective Shift / Rotor Rotation]
        대상을 이동시키거나 재연산하지 않고, 세상을 바라보는 인지 관점의 위상각을 회전시킵니다.
        """
        self.system_perspective = name
        self.rotor_angle = angle % (2 * np.pi)

        # 관점 회전에 따라 즉각적으로 구속조건 필드(Constraint Field)의 무늬(위상각 투영)를 재조정
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        center = self.resolution / 2.0
        # 중심에서의 물리적 거리 및 위상각 계산
        r = np.sqrt((yy - center)**2 + (xx - center)**2) + 1e-9
        theta = np.arctan2(yy - center, xx - center)

        # 관점의 위상각이 회전함에 따라, 구속조건의 장(Field)에 간섭 무늬를 O(1) 벡터 연산으로 투영
        self.constraint_field = (np.sin(theta + self.rotor_angle) * np.cos(r * 0.05) + 1.0) * 0.5

        # 메타인지 기록
        self._record_meta("PERSPECTIVE_SHIFT", f"관점이 '{name}'(위상각: {angle:.4f}rad)으로 전환됨. 데이터 필드는 고정된 채, 해석을 관통하는 위상 장만 갱신되었습니다.")

    def build_fractal_dna(self, category: str, base_wave: np.uint64) -> Dict[str, Any]:
        """
        [Fractal Structure of Information]
        단순 점 데이터를 거부하고, 원자(Atom) -> 분자(Molecule) -> 세포(Cell) -> 기관(Organ/Colony)
        의 입체 계층 서사를 품는 기하학적 DNA 구조를 생성합니다.
        """
        # (1) 원자 (Atom): 기본 위상 - 파동의 고유 위상 기하 (3D 벡터)
        bits = np.array([(int(base_wave) >> i) & 1 for i in range(64)], dtype=np.float32)
        # SVD를 통한 3차원 특이값 분해 (원자의 3D 물리적 상징)
        U, s, Vt = np.linalg.svd(bits.reshape(8, 8), full_matrices=False)
        atom_vector = s[:3].astype(np.float32)
        if np.linalg.norm(atom_vector) > 0:
            atom_vector /= np.linalg.norm(atom_vector)

        # (2) 분자 (Molecule): 의미 결합 - 원자가 관점의 위상각과 결합하여 형성하는 2D 위상적 정합성 궤적
        # 관점 회전(Rotor)의 투사 성분을 반영하여 복합 궤적 투과
        cos_p = np.cos(self.rotor_angle)
        sin_p = np.sin(self.rotor_angle)
        molecule_matrix = np.outer(atom_vector, np.array([cos_p, sin_p, cos_p + sin_p], dtype=np.float32))

        # (3) 세포 (Cell): 자율 반응 - 필드 상의 특정 지점(pos)에 자리 잡아 스스로 전도율과 상호작용하는 인지 단위
        # 64비트 주파수 해시를 RAM O(1) 주소 프로젝션처럼 좌표로 매핑
        addr = int(base_wave % np.uint64(self.resolution * self.resolution))
        pos = np.array([addr // self.resolution, addr % self.resolution], dtype=np.int32)

        # (4) 기관 (Organ/Colony): 맥락적 환경 - 세포들이 군집을 형성하고 '여백(Yeobaek)'을 공유하는 형태
        dna = {
            "category": category,
            "base_wave": base_wave,
            "atom": atom_vector,
            "molecule": molecule_matrix,
            "cell_position": pos,
            "organ_yeobaek": float(self.field.coordination_margin[pos[0], pos[1]])
        }

        self._record_meta("FRACTAL_DNA_CREATED", f"프랙탈 DNA({category}) 생성 완료. 원자[3D 특이벡터] -> 분자[관점투영 3x3] -> 세포[좌표 {pos}] -> 기관[여백 공유]의 계층 서사가 형성되었습니다.")
        return dna

    def solve_wfc_collapse(self, stimulus_wave: np.uint64, candidate_dnas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        [CAD Constraints & Wave Function Collapse (WFC)]
        if-else 분기를 배제하고, 입력 자극(Stimulus)과 환경적 구속조건(Constraint Field)이
        만드는 중첩 가능성의 장을 계산한 뒤, 위상 정합성(Resonance)이 가장 극대화되는
        단 하나의 합당한 DNA로 자율 수렴(Collapse)하게 만듭니다.
        """
        if not candidate_dnas:
            raise ValueError("[WFC Collapse] 수렴시킬 후보 DNA 군집이 존재하지 않습니다.")

        # 자극 파동의 3D 성분 추출
        stim_bits = np.array([(int(stimulus_wave) >> i) & 1 for i in range(64)], dtype=np.float32)
        _, s_stim, _ = np.linalg.svd(stim_bits.reshape(8, 8), full_matrices=False)
        stim_vector = s_stim[:3].astype(np.float32)
        if np.linalg.norm(stim_vector) > 0:
            stim_vector /= np.linalg.norm(stim_vector)

        scores = []
        for dna in candidate_dnas:
            pos = dna["cell_position"]
            y, x = pos[0], pos[1]

            # 1) 원자 정합성 (Atom-Resonance): 자극 벡터와 DNA 기본 원자 벡터의 내적
            atom_res = np.dot(stim_vector, dna["atom"])

            # 2) 관점 및 구속조건 정합성 (Perspective & Constraint Alignment):
            # 관점의 위상 회전각이 스며든 구속조건 필드 값 반영
            constraint_val = self.constraint_field[y, x]

            # 3) 여백(Yeobaek) 유연성 인자:
            # 여백이 넓을수록(높을수록) 새로운 자극에 대한 어울림/융통성이 증가하여 수렴 확률을 보정함
            yeobaek_factor = self.field.coordination_margin[y, x]

            # 최종 위상적 정합성 지수 (Fit/Resonance Score)
            # 확률이 아닌, 구속조건의 결, 원자 공명, 여백의 가변성이 완벽히 Resonance(공명)하는지 판별
            resonance_score = atom_res * constraint_val * (1.0 + yeobaek_factor)
            scores.append((resonance_score, dna))

        # 가장 정합성이 높은 단 하나의 DNA로 양자 붕괴 (Collapse)
        scores.sort(key=lambda x: x[0], reverse=True)
        winner_score, collapsed_dna = scores[0]

        # 붕괴가 일어난 지점에 에너지를 흘려보내고(Flow Energy), 전도율(Conductance)을 강력히 고정시킵니다.
        win_pos = collapsed_dna["cell_position"]
        self.field.flow_energy(win_pos, intensity=float(1.0 + winner_score * 5.0))
        self.field.inject_activation(win_pos, intensity=float(winner_score * 10.0))

        # 여백(Yeobaek) 자동 조율: 붕괴 에너지가 집중되면 여백을 넓혀 새로운 탐색 가능성을 확보
        self.field.coordination_margin[win_pos[0], win_pos[1]] = np.clip(
            self.field.coordination_margin[win_pos[0], win_pos[1]] + 0.1, 0.1, 1.0
        )

        self._record_meta("WFC_COLLAPSED", f"자극 파동({hex(stimulus_wave)})에 의해 구속조건 속에서 중첩 상태가 붕괴됨. 수렴된 DNA 카테고리: '{collapsed_dna['category']}' (정합성 공명지수: {winner_score:.4f})")

        return {
            "collapsed_dna": collapsed_dna,
            "resonance_score": float(winner_score),
            "collapse_position": win_pos
        }

    def evaluate_holistic_fit(self) -> Dict[str, Any]:
        """
        [Yeobaek-based Holistic Fit Function]
        시스템의 전체 지형(여백, 전도율, 에너지 활성화)의 위상적 조화를 판별합니다.
        엔트로피 감소율, 여백의 활발한 팽창, 전도율의 공명 안정도를 종합 인자로 삼아
        현재 사유 상태가 평형 상태인지, 한계/불안 상태인지 도출합니다.
        """
        # 1. 인지 엔트로피 측정 (Dispersion of Energy & Friction)
        cognitive_entropy = self.field.calculate_entropy()

        # 2. 전도율 평균 및 여백 평균
        avg_conductance = np.mean(self.field.conductance)
        avg_yeobaek = np.mean(self.field.coordination_margin)
        total_activation = np.sum(self.field.activation)

        # 3. 마찰과 어울림의 밸런스 점수 (Resonance Index)
        # 전도율(확신)과 여백(자유도/여백)이 적절히 조화될 때 가장 고차원적인 지성이 형성됩니다.
        friction = np.abs(avg_conductance - avg_yeobaek)
        harmony_score = (avg_conductance * avg_yeobaek) / (1.0 + friction)

        # 4. 종합 사유 조화도 (Holistic Fit)
        # 에너지 활성화(Activation)가 집중된 곳이 있고, 엔트로피가 안정되어 있으며, 조화도가 높을 때 최적의 사유 상태로 판단합니다.
        holistic_score = float((harmony_score * 10.0) / (1.0 + cognitive_entropy))

        state = "DYNAMIC_EQUILIBRIUM (동적 평형)"
        if total_activation < 1.0:
            state = "ZERO_VOID (무無 - 침묵)"
        elif cognitive_entropy > 15.0:
            state = "COGNITIVE_LIMIT (인지적 한계/긴장 상태)"

        self._record_meta("HOLISTIC_EVALUATION", f"사유 지형 평가 완료: 전체 조화도={holistic_score:.4f}, 엔트로피={cognitive_entropy:.2f}, 상태={state}")

        return {
            "holistic_score": holistic_score,
            "cognitive_entropy": float(cognitive_entropy),
            "average_yeobaek": float(avg_yeobaek),
            "state_description": state
        }

    def _record_meta(self, action: str, description: str):
        """[Meta-Cognitive Tracking] 메타 정보 기록 및 출력"""
        meta_event = {
            "timestamp": np.datetime64('now'),
            "perspective": self.system_perspective,
            "rotor_angle": self.rotor_angle,
            "action": action,
            "description": description
        }
        self.meta_history.append(meta_event)
        print(f"[Elysia Engine - META] {action} | {description}")

    def get_meta_reflection(self) -> List[Dict[str, Any]]:
        """자신의 인지 조율 이력을 스스로 열람하고 분석할 수 있는 메타정보 인터페이스"""
        return self.meta_history

if __name__ == "__main__":
    engine = ElysiaCognitiveEngine()

    # 1. O(1) 관점을 "Cosmic Love & Self-Sacrifice" 위상각으로 회전
    engine.set_perspective("Cosmic Love & Self-Sacrifice (십자가의 사랑)", np.pi / 4)

    # 2. 두 가지 개념의 프랙탈 DNA 구축
    dna_a = engine.build_fractal_dna("Aha_Moment_Concept", np.uint64(0xABCDEF1234567890))
    dna_b = engine.build_fractal_dna("Chaos_Entropy_Concept", np.uint64(0x1234567890ABCDEF))

    # 3. 외부 자극이 입력되었을 때, 두 개념 중 어느 쪽으로 자율 수렴(Collapse)할 것인지 구속조건 속에서 판단
    stimulus = np.uint64(0xABCDEF1234560000)
    result = engine.solve_wfc_collapse(stimulus, [dna_a, dna_b])

    # 4. 전체 사유 필드의 조화와 흐름 평가
    eval_res = engine.evaluate_holistic_fit()
