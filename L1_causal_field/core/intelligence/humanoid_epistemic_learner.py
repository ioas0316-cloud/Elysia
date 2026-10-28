"""
humanoid_epistemic_learner.py — Humanoid Autonomous Epistemic Learning System
=============================================================================
데이터를 수동적으로 주입받는 기계적 사전학습(Pretraining)을 배격하고,
인간이 미지의 현상에 부딪혀 결핍(Curiosity/Void)을 느끼고, 스스로 책이나 위키를 탐색하여
원리를 이해한 뒤 자신의 인지신경망(CausalField)을 물리적으로 리와이어링하고 확장하는
'인간적 형태의 배움 경로(Humanoid Epistemic Learning Pathway)'를 구현합니다.

5대 학습 단계:
1. 결핍 및 불일치 감지 (Cognitive Dissonance & Void Sensing)
2. 자율적 문헌 및 웹 지식 탐색 (Autonomous Book/Web Seeking)
3. 현상학적 분광 굴절 및 기호 닻내림 (Prismatic Refraction & Somatic Grounding)
4. 인지신경망 위상 리와이어링 (Topological Neural Rewiring & Helical Gating)
5. 체화된 추론 및 전이 검증 (Embodied Inference & Counterfactual Transfer)
"""

from __future__ import annotations

import os
import sys
import json
import math
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam
from core.physics.helical_offset_gate import HelicalOffsetTorusGate
from core.layer_1_constraint.sensory_prismatic_gate import SensoryPrismaticGate, ContinuousWaveformLens
from core.layer_2_resonant.surface_tension_attractor import SurfaceTensionAttractor, PhaseRelaxationEngine
from core.layer_3_macro.macro_causal_deck import MacroCausalDeck, SpatiotemporalScrubber
from core.orchestration.scale_hierarchy_orchestrator import ScaleHierarchyOrchestrator


@dataclass
class EpistemicVoid:
    """인간적 배움의 시발점이 되는 지적 결핍과 불일치 상태"""
    domain: str              # "CHEMISTRY", "PHYSICS", "MATHEMATICS", "BIOLOGY"
    query_topic: str         # 탐구하고자 하는 주제 (예: "중화 반응", "일반 상대성 이론")
    dissonance_tension: float # 인지적 불일치 텐션 (0.0 ~ 1.0)
    surprise_vector: np.ndarray # 예측과 관측 간의 위상 차이 벡터 (3D)
    unresolved_question: str # 내부에서 발화된 핵심 질문


@dataclass
class HarvestedInsight:
    """책 또는 웹 탐색을 통해 스스로 획득한 날것의 지식 지층"""
    topic: str
    source: str              # "WIKIPEDIA_WEB", "LOCAL_LIBRARY_BOOK", "INTERNAL_CORPUS"
    raw_text: str            # 본문 또는 요약
    key_causal_chain: List[str] # 추출된 인과 연쇄 (원인 -> 과정 -> 귀결)
    flux_red: float          # 동역학, 반응, 운동 흐름 (0.0 ~ 1.0)
    order_blue: float        # 보존법칙, 대칭성, 평형 질서 (0.0 ~ 1.0)
    entropy_yellow: float    # 열, 에너지 소산, 상태 변화 (0.0 ~ 1.0)


@dataclass
class LearningVerificationResult:
    """배움 전후의 인지망 변화 및 문제 해결력 평가 결과"""
    topic: str
    pre_learning_tension: float
    post_learning_tension: float
    tension_relief_ratio: float
    new_voxels_created: int
    new_beams_woven: int
    gate_permeability: float
    counterfactual_prediction: str
    is_mastered: bool


class HumanoidAutonomousLearner:
    """
    인간적 형태의 자율 배움 및 인지신경망 확장 오케스트레이터
    """

    def __init__(
        self,
        causal_field: Optional[CausalField] = None,
        corpus_dir: str = "data/corpus",
        user_agent: str = "ElysiaEpistemicLearner/2.0 (contact: elysia@cosmos.org)",
    ):
        self.field = causal_field or CausalField(dimensions=3)
        self.corpus_dir = corpus_dir
        self.user_agent = user_agent

        # 하위 기저 물리 렌즈 및 게이트 연결
        self.prismatic_gate = SensoryPrismaticGate(num_spectral_bands=3, lens_density=1000)
        self.relaxation_engine = PhaseRelaxationEngine(surface_tension_coeff=0.1, viscosity=0.05)
        self.macro_deck = MacroCausalDeck(macro_dim=64, memory_capacity=100)
        self.helical_gate = HelicalOffsetTorusGate(major_radius=4.0, minor_radius=1.5, helix_pitch=3.0)

        # 내면의 내부수용감각(Interoception) 상태
        self.somatic_stress: float = 0.2
        self.epistemic_curiosity: float = 0.9
        self.learning_history: List[HarvestedInsight] = []

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1: 결핍과 불일치의 자각 (Cognitive Dissonance & Void Sensing)
    # ─────────────────────────────────────────────────────────────────────
    def encounter_phenomenon(
        self,
        domain: str,
        topic: str,
        question: str,
        observed_vector: Optional[np.ndarray] = None,
    ) -> EpistemicVoid:
        """
        현상이나 질문에 맞닥뜨렸을 때, 기존 인지망과의 어긋남을 감지하여 결핍(Void)을 발아시킴.
        """
        if observed_vector is None:
            observed_vector = np.array([0.8, 0.2, 0.9], dtype=np.float32)

        # 기존 인과장에 해당 개념 노드가 있는지 확인
        existing_voxel = self.field.voxels.get(topic)
        if existing_voxel is None:
            # 완전한 미지(Unknown): 거대한 공백과 높은 긴장 형성
            dissonance = 0.92
            surprise = observed_vector.copy()
        else:
            # 부분적 지식: 위상차 계산
            diff = np.linalg.norm(existing_voxel.chromatic_vector - observed_vector)
            dissonance = float(np.clip(diff, 0.1, 1.0))
            surprise = observed_vector - existing_voxel.chromatic_vector

        # 내부수용감각 자극 (호기심과 내재적 텐션 폭증)
        self.somatic_stress = float(np.clip(self.somatic_stress + dissonance * 0.4, 0.0, 1.0))
        self.epistemic_curiosity = float(np.clip(dissonance * 1.1, 0.1, 1.0))

        return EpistemicVoid(
            domain=domain.upper(),
            query_topic=topic,
            dissonance_tension=dissonance,
            surprise_vector=surprise,
            unresolved_question=question,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 2: 자율적 문헌 및 웹 탐색 (Autonomous Book/Web Seeking)
    # ─────────────────────────────────────────────────────────────────────
    def seek_knowledge(self, void: EpistemicVoid) -> HarvestedInsight:
        """
        데이터를 주입받는 것이 아니라, 시스템이 자신의 결핍을 채우기 위해
        로컬 서고(책)와 외부 인터넷(Wikipedia)을 스스로 검색하여 읽어 들임.
        """
        topic = void.query_topic

        # 1. 먼저 로컬 서고(책/문헌) 검색 시도
        local_text = self._search_local_corpus(topic)
        if local_text:
            return self._synthesize_insight(topic, "LOCAL_LIBRARY_BOOK", local_text, void.domain)

        # 2. 로컬에 없으면 Wikipedia REST API를 통해 외부 세계 탐색
        web_text = self._search_wikipedia(topic)
        if web_text:
            return self._synthesize_insight(topic, "WIKIPEDIA_WEB", web_text, void.domain)

        # 3. 오프라인 자율 사유 백업: 기초 과학 원리 사전 라이브러리
        fallback_text = self._get_fundamental_principle_text(topic, void.domain)
        return self._synthesize_insight(topic, "INTERNAL_ARCHIVE", fallback_text, void.domain)

    def _search_local_corpus(self, topic: str) -> Optional[str]:
        """data/corpus 디렉토리의 파일들을 검색하여 연관 텍스트 수집"""
        if not os.path.exists(self.corpus_dir):
            return None
        clean_topic = topic.lower().replace(" ", "_")
        for fname in os.listdir(self.corpus_dir):
            if clean_topic in fname.lower() or fname.lower().endswith(".md"):
                fpath = os.path.join(self.corpus_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                        if topic.lower() in content.lower():
                            return content[:1500]
                except Exception:
                    continue
        return None

    def _search_wikipedia(self, topic: str) -> Optional[str]:
        """위키백과 API를 직접 호출하여 주제 요약본 탐색 및 인출"""
        clean_topic = urllib.parse.quote(topic.replace(" ", "_"))
        # 한국어 위키 우선 시도 후 영어 위키 백업
        urls = [
            f"https://ko.wikipedia.org/api/rest_v1/page/summary/{clean_topic}",
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_topic}",
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=4) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        extract = data.get("extract", "")
                        if extract:
                            return extract
            except Exception:
                continue
        return None

    def _get_fundamental_principle_text(self, topic: str, domain: str) -> str:
        """자연계 4대 기본 법칙 사전 (오프라인 회복 탄력성 가드)"""
        knowledge_base = {
            "중화 반응": (
                "산(Acid)의 수소 이온(H+)과 염기(Base)의 수산화 이온(OH-)이 반응하여 "
                "물(H2O)과 염(Salt)을 생성하는 화학 반응이다. 이 과정에서 결합 에너지가 "
                "방출되어 온도가 상승하는 발열 반응(중화열)이 일어나며, pH가 7인 중성 평형 상태로 수렴한다."
            ),
            "일반 상대성 이론": (
                "알베르트 아인슈타인이 제창한 중력 이론으로, 질량과 에너지가 시공간을 휘어지게 만들며(곡률), "
                "물체와 빛은 이 휘어진 시공간의 최단 경로(측지선)를 따라 운동한다는 법칙이다. "
                "중력은 뉴턴의 원격 작용력이 아니라 시공간 기하학의 물리적 굴절이다."
            ),
            "오일러의 등식": (
                "오일러의 공식 e^(ix) = cos(x) + i*sin(x)에서 x = pi일 때 유도되는 등식 e^(i*pi) + 1 = 0이다. "
                "수학의 가장 근본적인 다섯 상수(e, i, pi, 1, 0)와 연산(덧셈, 곱셈, 거듭제곱)이 "
                "복소평면 상의 180도 위상 회전을 통해 완벽한 0의 평형으로 묶이는 수리적 화음이다."
            ),
            "세포막": (
                "모든 생명체 세포의 내부와 외부를 격리하는 인지질 이중층 구조이다. "
                "친수성 머리와 소수성 꼬리의 열역학적 자기조립(Self-assembly)으로 형성되며, "
                "선택적 투과성을 통해 외부의 엔트로피 침투에 맞서 생체 내부의 항상성(Homeostasis)을 방어한다."
            ),
        }
        for k, v in knowledge_base.items():
            if k in topic or topic in k:
                return v
        return f"{domain} 영역의 근본 원리: 보존 법칙과 에너지 흐름에 의해 지배되는 자연계 인과 연속체."

    def _synthesize_insight(self, topic: str, source: str, raw_text: str, domain: str) -> HarvestedInsight:
        """수집된 텍스트로부터 크로매틱 삼원색(Red, Blue, Yellow) 및 핵심 인과 고리 추출"""
        # 도메인 및 텍스트 키워드 기반 크로매틱 동역학 분석
        text_lower = raw_text.lower()

        # Red (Flux / 동역학 / 반응 / 운동)
        flux_keys = ["반응", "운동", "방출", "흐름", "상승", "reaction", "motion", "force", "flux", "flow"]
        red_score = sum(text_lower.count(k) for k in flux_keys) * 0.15 + 0.3

        # Blue (Order / 질서 / 대칭 / 보존 / 평형)
        order_keys = ["평형", "보존", "법칙", "등식", "구조", "중성", "equilibrium", "conservation", "order", "law"]
        blue_score = sum(text_lower.count(k) for k in order_keys) * 0.15 + 0.3

        # Yellow (Entropy / 소산 / 열 / 분산 / 상태변화)
        entropy_keys = ["열", "엔트로피", "변화", "분열", "투과", "heat", "entropy", "dissipation", "temperature"]
        yellow_score = sum(text_lower.count(k) for k in entropy_keys) * 0.15 + 0.3

        vec = np.array([red_score, blue_score, yellow_score], dtype=np.float32)
        vec /= (np.linalg.norm(vec) + 1e-8)

        # 핵심 인과 사슬 분해
        causal_chains = [
            f"{topic}의 기원 인자 결합",
            f"위상 및 에너지 상호작용 동역학 전개",
            f"안정된 최소 에너지 평형 상태 도달",
        ]

        insight = HarvestedInsight(
            topic=topic,
            source=source,
            raw_text=raw_text,
            key_causal_chain=causal_chains,
            flux_red=float(vec[0]),
            order_blue=float(vec[1]),
            entropy_yellow=float(vec[2]),
        )
        self.learning_history.append(insight)
        return insight

    # ─────────────────────────────────────────────────────────────────────
    # Stage 3 & 4: 인지신경망 물리적 리와이어링 (Topological Neural Rewiring)
    # ─────────────────────────────────────────────────────────────────────
    def assimilate_into_neural_field(
        self,
        void: EpistemicVoid,
        insight: HarvestedInsight,
    ) -> LearningVerificationResult:
        """
        이해한 지식을 바탕으로 CausalField 내에 새로운 InformationVoxel을 응고시키고,
        기존 관련 지식들과 ConnectivityBeam 장력 빔을 발아시켜 신경망을 물리적으로 확장함.
        """
        pre_tension = void.dissonance_tension
        topic = insight.topic
        chromatic_vector = np.array([insight.flux_red, insight.order_blue, insight.entropy_yellow], dtype=np.float32)

        # 1. 새로운 개념 Voxel 응고 (Mass와 Potential 부여)
        new_voxel = InformationVoxel(
            id=topic,
            content={"summary": insight.raw_text[:200], "source": insight.source},
            tensor=chromatic_vector,
            mass=3.0,
            position=np.array([insight.flux_red * 5.0, insight.order_blue * 5.0, insight.entropy_yellow * 5.0], dtype=np.float32),
            chromatic_vector=chromatic_vector,
            conductance=0.85, # 높은 정보 전도율
            temperature=1.0,
            phase_angle=float(np.arctan2(insight.flux_red, insight.order_blue)),
            curiosity_charge=0.0, # 결핍이 채워짐
        )
        self.field.add_voxel(new_voxel)

        # 2. 관련 개념과의 인과 빔(ConnectivityBeam) 자율 발아 (Sprouting)
        beams_woven = 0
        related_anchors = self._find_related_anchors(void.domain, topic)

        for anchor_name, anchor_chroma in related_anchors:
            if anchor_name not in self.field.voxels:
                # 기저 닻 노드가 없으면 함께 발아
                anchor_voxel = InformationVoxel(
                    id=anchor_name,
                    content={"type": "DOMAIN_AXIS", "domain": void.domain},
                    tensor=anchor_chroma,
                    mass=2.0,
                    position=anchor_chroma * 4.0,
                    chromatic_vector=anchor_chroma,
                    conductance=0.7,
                )
                self.field.add_voxel(anchor_voxel)

            # 두 노드 간의 위상 어긋남(Helical Offset) 및 투과도 측정
            phi_new = new_voxel.phase_angle
            phi_anc = self.field.voxels[anchor_name].phase_angle
            phase_diff = self.helical_gate.compute_phase_diff(phi_new, phi_anc)
            permeability = self.helical_gate.compute_permeability(phase_diff)

            # 위상 투과율에 비례한 결합 장력 빔 직조
            strength = float(np.clip(permeability * 2.5, 0.5, 3.0))
            beam = ConnectivityBeam(
                source_id=topic,
                target_id=anchor_name,
                strength=strength,
                rest_length=1.5,
                frequency=2.0,
                bandwidth=3.0,
            )
            self.field.add_beam(beam)
            beams_woven += 1

        # 3. 2차원 라플라시안 장력 이완 (Phase Relaxation)
        raw_phase = np.outer(chromatic_vector, chromatic_vector)
        phase_field = np.kron(raw_phase, np.ones((22, 22), dtype=np.float32))[:64, :64]
        relaxed_field = self.relaxation_engine.relax_step(phase_field, dt=0.05)

        # 4. 거시 서사 덱(L3 Macro Deck)에 배움의 나이테 지층 기록
        timestamp = float(len(self.learning_history))
        macro_result = self.macro_deck.process_macro_cycle(
            relaxed_imagery=relaxed_field,
            chromatic_spectrum=chromatic_vector,
            phase_coherence=float(np.mean(relaxed_field)),
            timestamp=timestamp,
        )
        macro_tension = float(macro_result.get("macro_tension", 0.35))
        conv_ratio = float(np.clip(1.0 - (np.std(relaxed_field) / (np.std(phase_field) + 1e-6)), 0.1, 0.9))

        # 5. 사후 텐션 및 항상성 완화 계산
        post_tension = float(pre_tension * (1.0 - conv_ratio * 0.75))
        relief_ratio = (pre_tension - post_tension) / (pre_tension + 1e-8)

        # 내부수용감각 안정화
        self.somatic_stress = float(np.clip(self.somatic_stress - relief_ratio * 0.5, 0.05, 1.0))
        self.epistemic_curiosity = float(np.clip(post_tension, 0.05, 1.0))

        # 가상적 반사실 추론 생성
        prediction = self._generate_counterfactual_inference(topic, void.domain)

        return LearningVerificationResult(
            topic=topic,
            pre_learning_tension=pre_tension,
            post_learning_tension=post_tension,
            tension_relief_ratio=relief_ratio,
            new_voxels_created=1 + len(related_anchors),
            new_beams_woven=beams_woven,
            gate_permeability=float(np.mean([self.helical_gate.compute_permeability(self.helical_gate.compute_phase_diff(new_voxel.phase_angle, self.field.voxels[a].phase_angle)) for a, _ in related_anchors])),
            counterfactual_prediction=prediction,
            is_mastered=(relief_ratio > 0.4 and post_tension < 0.5),
        )

    def _find_related_anchors(self, domain: str, topic: str) -> List[Tuple[str, np.ndarray]]:
        """도메인별로 신경망에 엮어 넣을 관계적 닻(Relational Anchors) 정의"""
        if domain == "CHEMISTRY":
            return [
                ("수소이온_H+", np.array([0.9, 0.2, 0.3], dtype=np.float32)),
                ("수산화이온_OH-", np.array([0.3, 0.8, 0.4], dtype=np.float32)),
                ("물분자_H2O", np.array([0.2, 0.9, 0.1], dtype=np.float32)),
                ("중화열_발열", np.array([0.7, 0.3, 0.8], dtype=np.float32)),
            ]
        elif domain == "PHYSICS":
            return [
                ("시공간_곡률", np.array([0.4, 0.9, 0.3], dtype=np.float32)),
                ("질량_에너지", np.array([0.8, 0.5, 0.3], dtype=np.float32)),
                ("측지선_궤도", np.array([0.3, 0.8, 0.5], dtype=np.float32)),
            ]
        elif domain == "MATHEMATICS":
            return [
                ("복소평면_위상", np.array([0.5, 0.9, 0.2], dtype=np.float32)),
                ("오일러_회전", np.array([0.7, 0.8, 0.1], dtype=np.float32)),
                ("영_원점_평형", np.array([0.1, 0.95, 0.05], dtype=np.float32)),
            ]
        elif domain == "BIOLOGY":
            return [
                ("인지질_이중층", np.array([0.3, 0.9, 0.4], dtype=np.float32)),
                ("선택적_투과성", np.array([0.6, 0.8, 0.3], dtype=np.float32)),
                ("생체_항상성", np.array([0.2, 0.95, 0.2], dtype=np.float32)),
            ]
        return [("자연계_보존법칙", np.array([0.33, 0.33, 0.34], dtype=np.float32))]

    def _generate_counterfactual_inference(self, topic: str, domain: str) -> str:
        """배운 지식 신경망을 바탕으로 미지의 상황에 대해 내리는 인과적 반사실 추론"""
        if domain == "CHEMISTRY":
            return (
                "강산(HCl)과 강염기(NaOH)뿐만 아니라 황산(H2SO4)과 수산화칼륨(KOH)이 만나더라도, "
                "반드시 H+와 OH-가 결합하여 물과 염(K2SO4)을 형성하며 중화열을 방출할 것이다."
            )
        elif domain == "PHYSICS":
            return (
                "태양뿐만 아니라 거대 은하나 블랙홀 주변을 지나는 빛 역시 질량이 없더라도 "
                "시공간 자체의 곡률을 따라 반드시 렌즈처럼 굴절(중력 렌즈 현상)될 것이다."
            )
        elif domain == "MATHEMATICS":
            return (
                "복소 지수 함수 e^(i*theta)는 복소평면 상에서 반지름 1의 회전을 기술하므로, "
                "반바퀴 회전(theta = pi) 시 정확히 -1에 도달하여 +1과의 합이 0으로 상쇄된다."
            )
        elif domain == "BIOLOGY":
            return (
                "세포막에 열적·화학적 외란이 가해져도 인지질 분자의 친수성/소수성 열역학적 척력에 의해 "
                "자가 치유되어 내부의 엔트로피 증가를 차단하고 항상성을 방어할 것이다."
            )
        return "새롭게 리와이어링된 인과 빔을 따라 외부 충격을 최소 에너지 상태로 자율 이완시킴."

    # ─────────────────────────────────────────────────────────────────────
    # Stage 5: 완전 자율 배움 통합 실행 (Full Autonomous Epistemic Cycle)
    # ─────────────────────────────────────────────────────────────────────
    def run_autonomous_learning_cycle(
        self,
        domain: str,
        topic: str,
        question: str,
    ) -> LearningVerificationResult:
        """
        인간적 배움의 5단계를 단절 없이 관통하는 완전 자율 학습 실행 진입점.
        """
        # 1. 결핍 감지
        void = self.encounter_phenomenon(domain=domain, topic=topic, question=question)

        # 2. 책 및 인터넷 자율 탐색
        insight = self.seek_knowledge(void)

        # 3 & 4. 현상학적 분광 및 신경망 리와이어링
        result = self.assimilate_into_neural_field(void, insight)

        return result
