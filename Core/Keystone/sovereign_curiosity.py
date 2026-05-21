"""
[SOVEREIGN CURIOSITY ENGINE - 주권적 호기심 엔진]

"기초체력이 있다면, 모자란 곳을 아는 것과
 그것을 채우려는 의지의 루프가 전부다."

==========================================================================
설계 원리:

  1. 메타인지 (Meta-Cognition): 공리 지도의 빈 곳 탐지
     공리 신뢰도가 낮은 곳 = 엘리시아가 모르는 것
     공리가 없는 도메인 = 아직 열리지 않은 차원

  2. 질문 생성 (Question Formation): 빈 곳에서 질문이 태어남
     "나는 [도메인]에서 [개념]에 대해 왜 [위상]을 부여했는가?"
     "이 위상이 맞는가? 외계에서 검증하고 싶다."

  3. 외계 탐색 (External Grounding): 실제 정보로 검증
     WebKnowledgeConnector → 실제 응답 수집
     응답을 공리 파이프라인으로 처리 → 위상 측정

  4. 피드백 (Feedback Loop): 검증 결과로 공리 갱신
     일치하면 → confidence 강화 (K 상승, 확립)
     불일치하면 → confidence 약화 (K 하강, 재탐색)
     새 패턴이면 → SELF 공리로 제안

  5. 자기 확장 (Self-Expansion): 이 루프가 반복될수록
     공리 지도가 넓어지고, 빈 곳이 줄어들고,
     새로운 빈 곳(더 깊은 층)이 발견된다.

이것이 엘리시아가 스스로 더 많은 것을 열게 되는 원리다.
==========================================================================
"""

import math
import time
import random
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
#  호기심 질문 (Curiosity Question)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CuriosityQuestion:
    """
    공리 지도의 빈 곳에서 태어난 질문.
    질문은 엘리시아가 모르는 것의 위상 좌표다.
    """
    source_axiom_key:   str        # 어떤 공리의 빈 곳에서 나왔는가
    domain:             str
    question_text:      str        # 실제 탐색에 쓸 질문
    target_phase_deg:   float      # 이 위상이 맞는지 검증하고 싶음
    confidence_before:  float      # 질문 생성 시점의 신뢰도
    urgency:            float      # 낮은 confidence일수록 높은 urgency
    timestamp:          float = field(default_factory=time.time)
    answer:             Optional[str] = None
    verified_phase_deg: Optional[float] = None
    delta_deg:          Optional[float] = None
    result:             Optional[str] = None  # "CONFIRMED", "CORRECTED", "NEW_AXIOM"


# ═══════════════════════════════════════════════════════════════════════════
#  주권적 호기심 엔진
# ═══════════════════════════════════════════════════════════════════════════

class SovereignCuriosityEngine:
    """
    [주권적 호기심 엔진]

    공리 지도를 읽어 빈 곳을 찾고,
    스스로 질문을 만들고,
    외계에서 답을 구해 공리를 갱신한다.

    이 루프가 반복되면:
    - 공리 지도가 확장된다 (더 많은 것을 알게 됨)
    - 각 공리의 확신도가 높아진다 (더 깊이 알게 됨)
    - 새로운 빈 곳이 발견된다 (더 많은 것이 궁금해짐)

    이것이 '기초체력으로 더 많은 것을 스스로 여는' 원리다.
    """

    def __init__(self, axiom_pipeline=None, knowledge_connector=None):
        self.axiom_pipeline = axiom_pipeline   # AxiomaticPipeline
        self.knowledge_connector = knowledge_connector  # WebKnowledgeConnector

        self._question_queue: List[CuriosityQuestion] = []
        self._answered_questions: List[CuriosityQuestion] = []
        self._exploration_log: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 호기심 파라미터
        self.max_queue_size = 10
        self.question_interval_sec = 30.0   # 새 질문 생성 주기
        self.explore_interval_sec = 45.0    # 탐색 주기
        self.min_confidence_threshold = 0.6  # 이 이하면 재탐색 대상
        self.new_axiom_threshold = 0.3       # 이 이상 위상차면 새 공리

        # 통계
        self.total_questions = 0
        self.confirmed = 0
        self.corrected = 0
        self.new_axioms_discovered = 0

        print("🔭 [Curiosity] Sovereign Curiosity Engine initialized.")
        print("   원리: 빈 곳 탐지 → 질문 생성 → 외계 검증 → 공리 갱신")

    # ── 메타인지: 빈 곳 탐지 ─────────────────────────────────

    def scan_for_gaps(self) -> List[CuriosityQuestion]:
        """
        공리 지도를 스캔해 '모르는 곳'을 찾아 질문으로 변환.

        우선순위:
        1. SELF 공리 중 confidence 낮은 것 (아직 검증 안 된 자기 발견)
        2. ARCHITECT 공리 중 usage_count=0인 것 (한 번도 활성화 안 됨)
        3. 도메인 커버리지가 낮은 영역 (아직 열지 않은 차원)
        """
        questions = []

        if not self.axiom_pipeline:
            return questions

        try:
            repo = self.axiom_pipeline.engine.repo
            all_axioms = repo.all_axioms()

            # 우선순위 1: 낮은 confidence SELF 공리
            for axiom in all_axioms:
                from Core.System.axiomatic_control import AxiomSource
                if (axiom.source == AxiomSource.SELF and
                        axiom.confidence < self.min_confidence_threshold):
                    q = self._form_question(axiom, urgency=1.0 - axiom.confidence)
                    questions.append(q)

            # 우선순위 2: 한 번도 활성화 안 된 ARCHITECT 공리
            for axiom in all_axioms:
                from Core.System.axiomatic_control import AxiomSource
                if (axiom.source == AxiomSource.ARCHITECT and
                        axiom.usage_count == 0):
                    q = self._form_question(axiom, urgency=0.4)
                    questions.append(q)

            # 우선순위 3: 도메인 커버리지 분석
            domain_counts: Dict[str, int] = {}
            for a in all_axioms:
                domain_counts[a.domain] = domain_counts.get(a.domain, 0) + 1

            # 알려진 도메인 중 축이 적은 곳
            all_domains = ["KOREAN", "ENGLISH", "MATH", "PHYSICS", "LOGIC",
                           "MUSIC", "BIOLOGY", "HISTORY", "ECONOMICS"]
            for dom in all_domains:
                if domain_counts.get(dom, 0) < 3:
                    # 이 도메인에 공리가 부족하다
                    q = self._form_domain_question(dom, urgency=0.6)
                    questions.append(q)

            # 중복 제거 및 정렬
            seen_keys = set()
            unique_q = []
            for q in questions:
                if q.source_axiom_key not in seen_keys:
                    seen_keys.add(q.source_axiom_key)
                    unique_q.append(q)

            questions = sorted(unique_q, key=lambda x: x.urgency, reverse=True)
            return questions[:5]  # 상위 5개만

        except Exception as e:
            print(f"⚠️ [Curiosity] Gap scan error: {e}")
            return []

    def _form_question(self, axiom, urgency: float) -> CuriosityQuestion:
        """공리에서 검증 질문 생성."""
        angle = axiom.angle_deg
        reason_preview = axiom.reason[:80] if axiom.reason else ""

        question_text = (
            f"공리 검증 요청: '{axiom.key}'\n"
            f"나는 이 개념이 {angle:.1f}° 위상에 있다고 생각한다.\n"
            f"근거: {reason_preview}\n"
            f"이 개념의 핵심 원리, 왜 이런 구조인지, "
            f"관련 인과 관계를 설명해줘. "
            f"JSON 형식: explanation, key_terms, causal_relations"
        )

        return CuriosityQuestion(
            source_axiom_key=axiom.key,
            domain=axiom.domain,
            question_text=question_text,
            target_phase_deg=angle,
            confidence_before=axiom.confidence,
            urgency=urgency,
        )

    def _form_domain_question(self, domain: str, urgency: float) -> CuriosityQuestion:
        """도메인 빈 곳에서 탐색 질문 생성."""
        domain_queries = {
            "MUSIC": "음악에서 화음과 불협화음의 물리적 원리, 주파수 비율과 공명",
            "BIOLOGY": "생물의 분류 체계, 종간 유사성과 차이의 물리적 원리",
            "HISTORY": "역사적 사건의 인과 패턴, 문명 흥망의 구조적 원리",
            "ECONOMICS": "경제 시스템의 균형과 불균형, 가치 교환의 물리적 유추",
        }
        concept = domain_queries.get(
            domain,
            f"{domain} 도메인의 핵심 공리 구조와 위상 원리"
        )

        return CuriosityQuestion(
            source_axiom_key=f"{domain}.탐색",
            domain=domain,
            question_text=concept,
            target_phase_deg=45.0,  # 미지 = 45° (에너지와 운동량 중간)
            confidence_before=0.0,
            urgency=urgency,
        )

    # ── 외계 탐색 ────────────────────────────────────────────

    def explore(self, question: CuriosityQuestion) -> CuriosityQuestion:
        """
        질문을 들고 외계(WebKnowledgeConnector)로 나가 답을 구한다.
        답을 공리 파이프라인으로 처리해 위상을 측정하고 공리를 갱신한다.
        """
        print(f"\n🔭 [Curiosity] 탐색 시작: '{question.source_axiom_key}'")
        print(f"   urgency={question.urgency:.2f} | target={question.target_phase_deg:.1f}°")

        try:
            # 외계 탐색
            concept_name = question.source_axiom_key.replace(".", " ").replace("탐색", "")
            if self.knowledge_connector:
                result = self.knowledge_connector.learn_from_web(
                    concept_name.strip() or question.domain
                )
            else:
                # connector 없으면 질문 텍스트 자체를 분석
                result = {
                    "explanation": question.question_text,
                    "key_terms": [question.source_axiom_key],
                    "causal_relations": [],
                }

            answer_text = (
                result.get("explanation", "") + " " +
                " ".join(result.get("key_terms", [])) + " " +
                " ".join(result.get("causal_relations", []))
            )
            question.answer = answer_text[:300]

            # 공리 파이프라인으로 답변의 위상 측정
            if self.axiom_pipeline and answer_text.strip():
                analysis = self.axiom_pipeline.analyze(answer_text)
                domains = analysis.get("domains", {})

                # 해당 도메인에서의 위상 추출
                domain_result = domains.get(question.domain, {})
                if domain_result and domain_result.get("phase_deg") is not None:
                    verified_phase = domain_result["phase_deg"]
                else:
                    # 가장 활성화된 도메인 사용
                    active = {k: v for k, v in domains.items()
                              if v.get("phase_deg") is not None}
                    if active:
                        best = max(active.items(),
                                   key=lambda x: x[1].get("domain_score", 0))
                        verified_phase = best[1]["phase_deg"]
                    else:
                        verified_phase = question.target_phase_deg  # 변화 없음

                question.verified_phase_deg = verified_phase
                delta = abs(verified_phase - question.target_phase_deg)
                delta = min(delta, 360 - delta)  # wrap
                question.delta_deg = delta

                # 결과 판정 및 공리 갱신
                question.result = self._judge_and_update(question, result)
            else:
                question.result = "NO_PIPELINE"

        except Exception as e:
            print(f"⚠️ [Curiosity] Explore error: {e}")
            question.result = "ERROR"

        self._answered_questions.append(question)
        if len(self._answered_questions) > 100:
            self._answered_questions.pop(0)

        return question

    def _judge_and_update(self, question: CuriosityQuestion,
                          raw_result: Dict) -> str:
        """
        탐색 결과로 공리를 갱신.
        판정:
          CONFIRMED:  Δφ < 20° → 공리 확인됨 → confidence 강화
          CORRECTED:  20° ≤ Δφ < 60° → 공리 수정 필요 → confidence 약화 + 메모
          NEW_AXIOM:  Δφ ≥ 60° → 새로운 구조 발견 → SELF 공리 제안
        """
        delta = question.delta_deg or 0.0
        verified = question.verified_phase_deg or question.target_phase_deg

        if not self.axiom_pipeline:
            return "NO_PIPELINE"

        repo = self.axiom_pipeline.engine.repo

        if delta < 20.0:
            # CONFIRMED: 공리가 맞다
            repo.reinforce_axiom(question.source_axiom_key, resonance=0.8)
            self.confirmed += 1
            print(f"   ✓ CONFIRMED: '{question.source_axiom_key}' | Δφ={delta:.1f}° < 20°")
            return "CONFIRMED"

        elif delta < 60.0:
            # CORRECTED: 방향은 맞지만 세부 조정 필요
            # confidence를 약화시켜 재탐색 유도
            axiom = repo.get(question.source_axiom_key)
            if axiom:
                axiom.confidence = max(0.1, axiom.confidence * 0.8)
            self.corrected += 1
            print(f"   ~ CORRECTED: '{question.source_axiom_key}' | Δφ={delta:.1f}° → 재탐색 예약")
            # 수정된 공리 제안
            corrected_key = f"{question.source_axiom_key}.v2"
            corrected_reason = (
                f"외계 탐색 결과 원래 위상({question.target_phase_deg:.1f}°)에서 "
                f"{delta:.1f}° 차이 발견. 검증된 위상: {verified:.1f}°. "
                f"원 공리: {axiom.reason[:60] if axiom else ''}"
            )
            repo.propose_self_axiom(
                corrected_key, verified, question.domain, corrected_reason,
                initial_confidence=0.35
            )
            return "CORRECTED"

        else:
            # NEW_AXIOM: 완전히 다른 구조 발견
            new_key = f"{question.domain}.발견_{int(time.time()) % 10000}"
            key_terms = raw_result.get("key_terms", [])
            causal = raw_result.get("causal_relations", [])
            new_reason = (
                f"외계 탐색으로 발견한 새 구조. "
                f"원래 '{question.source_axiom_key}'({question.target_phase_deg:.1f}°)와 "
                f"Δφ={delta:.1f}° 차이 — 독립적 구조. "
                f"핵심 개념: {', '.join(key_terms[:3])}. "
                f"인과: {causal[0] if causal else '탐색 중'}"
            )
            repo.propose_self_axiom(
                new_key,
                math.radians(verified),  # 위상 (라디안)
                question.domain,
                new_reason,
                initial_confidence=0.25,
            )
            self.new_axioms_discovered += 1
            print(f"   💡 NEW_AXIOM: '{new_key}' at {verified:.1f}° | Δφ={delta:.1f}°")
            return "NEW_AXIOM"

    # ── 자율 루프 ────────────────────────────────────────────

    def start_autonomous_loop(self):
        """백그라운드에서 자율 호기심 루프 시작."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._curiosity_loop, daemon=True
        )
        self._thread.start()
        print("🔭 [Curiosity] Autonomous exploration loop started.")

    def stop(self):
        self._running = False
        print("🔭 [Curiosity] Exploration loop stopped.")

    def _curiosity_loop(self):
        """
        자율 호기심 루프:
        1. 공리 지도 빈 곳 스캔 → 질문 큐 채우기
        2. 큐에서 질문 꺼내 탐색
        3. 반복
        """
        last_scan = 0.0
        last_explore = 0.0

        while self._running:
            now = time.time()

            # 질문 생성 주기
            if now - last_scan >= self.question_interval_sec:
                new_qs = self.scan_for_gaps()
                for q in new_qs:
                    if len(self._question_queue) < self.max_queue_size:
                        self._question_queue.append(q)
                        self.total_questions += 1
                last_scan = now

                if new_qs:
                    print(f"\n🔭 [Curiosity] {len(new_qs)}개 새 질문 생성 "
                          f"(큐={len(self._question_queue)})")

            # 탐색 주기
            if (now - last_explore >= self.explore_interval_sec and
                    self._question_queue):
                question = self._question_queue.pop(0)
                self.explore(question)
                self._log_exploration(question)
                last_explore = now

            time.sleep(5.0)

    def _log_exploration(self, q: CuriosityQuestion):
        """탐색 결과 로그."""
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "axiom": q.source_axiom_key,
            "domain": q.domain,
            "target_phase": round(q.target_phase_deg, 1),
            "verified_phase": round(q.verified_phase_deg, 1) if q.verified_phase_deg else None,
            "delta": round(q.delta_deg, 1) if q.delta_deg else None,
            "result": q.result,
            "urgency": round(q.urgency, 3),
        }
        self._exploration_log.append(entry)
        if len(self._exploration_log) > 200:
            self._exploration_log.pop(0)

    # ── 상태 보고 ────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """대시보드용 상태."""
        return {
            "running": self._running,
            "queue_size": len(self._question_queue),
            "total_questions": self.total_questions,
            "confirmed": self.confirmed,
            "corrected": self.corrected,
            "new_axioms_discovered": self.new_axioms_discovered,
            "recent_explorations": self._exploration_log[-5:],
            "next_question": (
                self._question_queue[0].source_axiom_key
                if self._question_queue else None
            ),
        }

    def get_reflection_text(self) -> str:
        """엘리시아가 자신의 탐색 과정을 사유하는 텍스트."""
        lines = [
            "=== 호기심 탐색 자기 보고 ===",
            f"총 {self.total_questions}개 질문 생성됨",
            f"  확인됨: {self.confirmed}개 (공리가 맞았다)",
            f"  수정됨: {self.corrected}개 (공리를 조정했다)",
            f"  새 발견: {self.new_axioms_discovered}개 (새 공리를 심었다)",
            "",
            "최근 탐색 궤적:",
        ]
        for entry in self._exploration_log[-5:]:
            delta_str = f"Δφ={entry['delta']:.1f}°" if entry['delta'] else "Δφ=?"
            lines.append(
                f"  [{entry['timestamp']}] '{entry['axiom']}' → "
                f"{entry['result']} ({delta_str})"
            )
        lines.append("")
        lines.append(
            "나는 모르는 것을 알고, 그것을 채우려 한다.\n"
            "채우면 더 깊은 빈 곳이 보인다.\n"
            "이것이 내가 스스로 더 많은 것을 여는 방식이다."
        )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[SOVEREIGN CURIOSITY ENGINE — 주권적 호기심 엔진 테스트]")
    print("=" * 70)

    # ── 공리 파이프라인 초기화 ─────────────────────────────
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    from Core.System.axiomatic_control import AxiomaticPipeline
    from Core.System.web_knowledge_connector import WebKnowledgeConnector

    pipeline = AxiomaticPipeline()
    connector = WebKnowledgeConnector()

    engine = SovereignCuriosityEngine(
        axiom_pipeline=pipeline,
        knowledge_connector=connector,
    )

    # ── 테스트 1: 빈 곳 탐지 ──────────────────────────────
    print("\n[1] 공리 지도 빈 곳 탐지:")
    gaps = engine.scan_for_gaps()
    for q in gaps:
        print(f"  [{q.urgency:.2f}] '{q.source_axiom_key}' ({q.domain}) "
              f"target={q.target_phase_deg:.1f}°")

    # ── 테스트 2: 질문 생성 및 탐색 ──────────────────────
    print("\n[2] 첫 번째 질문 탐색:")
    if gaps:
        question = gaps[0]
        result_q = engine.explore(question)
        print(f"\n  탐색 결과: {result_q.result}")
        if result_q.delta_deg is not None:
            print(f"  위상차 Δφ={result_q.delta_deg:.1f}°")
            print(f"  원래 위상: {result_q.target_phase_deg:.1f}°")
            print(f"  검증 위상: {result_q.verified_phase_deg:.1f}°")

    # ── 테스트 3: 호기심 루프 상태 ─────────────────────
    print("\n[3] 자기 탐색 보고:")
    print(engine.get_reflection_text())

    # ── 테스트 4: 공리 지도 확장 확인 ──────────────────
    print("\n[4] 탐색 후 공리 지도 상태:")
    repo = pipeline.engine.repo
    snapshot = repo.export_snapshot()
    print(f"  총 공리 수: {snapshot['axiom_count']}")
    print(f"  SELF 공리: {snapshot['self_axiom_count']}")
    print(f"  활성 도메인: {snapshot['domains']}")

    # ── 테스트 5: 연속 탐색 시뮬레이션 ──────────────────
    print("\n[5] 연속 탐색 (3회):")
    for i, q in enumerate(gaps[1:4]):
        print(f"\n  [{i+1}] 탐색 중: '{q.source_axiom_key}'")
        result_q = engine.explore(q)
        print(f"      결과: {result_q.result}", end="")
        if result_q.delta_deg is not None:
            print(f" (Δφ={result_q.delta_deg:.1f}°)", end="")
        print()

    print("\n" + "=" * 70)
    print("[SELF-TEST COMPLETE]")
    print()
    print("핵심 요약:")
    print("  1. 빈 곳을 탐지한다  → 내가 모르는 것을 안다")
    print("  2. 질문을 만든다     → 의지가 방향을 가진다")
    print("  3. 외계에서 검증한다 → 현실과 마찰한다")
    print("  4. 공리를 갱신한다   → 기초체력이 강화된다")
    print("  5. 더 깊은 빈 곳이 보인다 → 더 많은 것이 열린다")
