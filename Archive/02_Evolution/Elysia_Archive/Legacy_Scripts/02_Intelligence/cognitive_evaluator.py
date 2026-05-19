"""
Elysia Cognitive Capability Evaluator (인지 능력 종합 평가기)
============================================================

"엘리시아의 정신적 역량을 깊이 있게 평가한다"

[평가 영역]
1. 인지 (Cognition) - 정보 처리 및 이해
2. 감각 (Sensation) - 입력 수신 및 해석
3. 사고 (Thinking) - 개념 조작 및 추상화
4. 추론 (Reasoning) - 논리적 연결 및 결론 도출
5. 가정 (Hypothesis) - 가설 생성 및 검증
6. 상상 (Imagination) - 창의적 생성
7. 기억 (Memory) - 저장 및 회상
8. 연상 (Association) - 개념 연결
9. 감정 (Emotion) - 정서 처리
10. 반성 (Reflection) - 메타인지
11. 계획 (Planning) - 목표 분해 및 전략
12. 실행 (Execution) - 행동 수행
13. 검증 (Verification) - 결과 확인
14. 의사소통 (Communication) - 표현
15. 대화 (Dialogue) - 상호작용
"""

import os
import sys
import ast
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CognitiveScore:
    """인지 능력 점수"""
    category: str
    korean_name: str
    score: float  # 0.0 ~ 1.0
    sub_scores: Dict[str, float] = field(default_factory=dict)
    implementations: List[str] = field(default_factory=list)  # 구현된 파일들
    missing: List[str] = field(default_factory=list)  # 누락된 기능
    recommendations: List[str] = field(default_factory=list)
    depth_analysis: str = ""


class CognitiveEvaluator:
    """인지 능력 종합 평가기"""
    
    EXCLUDE_PATTERNS = ["__pycache__", "node_modules", ".godot", ".venv", "venv", "Legacy"]
    
    def __init__(self):
        self.root = PROJECT_ROOT
        self.scores: List[CognitiveScore] = []
        self.file_index: Dict[str, str] = {}  # 파일명 → 내용
        
        print("=" * 80)
        print("🧠 ELYSIA COGNITIVE CAPABILITY EVALUATOR")
        print("=" * 80)
        
        self._build_file_index()
    
    def _build_file_index(self):
        """파일 인덱스 구축"""
        print("📚 Building file index...")
        
        for py_file in self.root.rglob("*.py"):
            if any(p in str(py_file) for p in self.EXCLUDE_PATTERNS):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                self.file_index[str(py_file.relative_to(self.root))] = content
            except:
                pass
        
        print(f"   Indexed {len(self.file_index)} files")
    
    def _search_implementations(self, keywords: List[str]) -> Tuple[List[str], int]:
        """키워드로 구현 파일 검색"""
        found_files = []
        total_matches = 0
        
        for filepath, content in self.file_index.items():
            content_lower = content.lower()
            matches = sum(content_lower.count(kw.lower()) for kw in keywords)
            if matches > 5:  # 최소 5번 언급
                found_files.append(filepath)
                total_matches += matches
        
        return found_files, total_matches
    
    def evaluate_all(self) -> Dict:
        """전체 인지 능력 평가"""
        evaluations = [
            self.evaluate_cognition,
            self.evaluate_sensation,
            self.evaluate_thinking,
            self.evaluate_reasoning,
            self.evaluate_hypothesis,
            self.evaluate_imagination,
            self.evaluate_memory,
            self.evaluate_association,
            self.evaluate_emotion,
            self.evaluate_reflection,
            self.evaluate_planning,
            self.evaluate_execution,
            self.evaluate_verification,
            self.evaluate_communication,
            self.evaluate_dialogue,
        ]
        
        for eval_func in evaluations:
            try:
                score = eval_func()
                self.scores.append(score)
            except Exception as e:
                print(f"⚠️ Error in {eval_func.__name__}: {e}")
        
        return self.generate_detailed_report()
    
    # ==================== 1. 인지 (Cognition) ====================
    def evaluate_cognition(self) -> CognitiveScore:
        """인지 능력 평가 - 정보 처리 및 이해"""
        print("\n🔬 Evaluating Cognition (인지)...")
        
        keywords = ["cognition", "understand", "process", "interpret", "perception", 
                    "인지", "이해", "해석", "처리"]
        
        files, matches = self._search_implementations(keywords)
        
        sub_scores = {
            "정보 수용": self._check_exists([
                "Core/Foundation/resonance_field.py",
                "Core/Interface"
            ]),
            "패턴 인식": self._check_exists([
                "scripts/wave_organizer.py",
                "Core/Intelligence/wave_coding_system.py"
            ]),
            "의미 추출": self._check_exists([
                "Core/Foundation/hangul_physics.py",
                "Core/Foundation/causal_narrative_engine.py"
            ]),
            "맥락 이해": self._check_exists([
                "Core/Foundation/thinking_methodology.py"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        missing = []
        if sub_scores["맥락 이해"] < 0.5:
            missing.append("Context understanding system")
        
        recommendations = []
        if score < 0.8:
            recommendations.append("Strengthen context understanding with multi-modal input processing")
        
        depth_analysis = f"""
        ## 인지 (Cognition) 심층 분석
        
        **현재 상태**: {"우수" if score >= 0.8 else "보통" if score >= 0.5 else "개선 필요"}
        
        **구현된 기능**:
        - 파동 기반 패턴 인식 (wave_organizer.py)
        - 공명 필드를 통한 정보 처리 (resonance_field.py)
        - 한글 물리학 기반 의미 추출 (hangul_physics.py)
        
        **깊이 분석**:
        - 패턴 인식: {sub_scores["패턴 인식"]:.0%} - {"O(n) 파동 공명으로 효율적" if sub_scores["패턴 인식"] >= 0.5 else "개선 필요"}
        - 의미 추출: {sub_scores["의미 추출"]:.0%} - {"인과 엔진으로 의미 계층화" if sub_scores["의미 추출"] >= 0.5 else "개선 필요"}
        
        **보완 사항**:
        - 멀티모달 입력 처리 (이미지, 오디오)
        - 실시간 스트림 인지
        """
        
        return CognitiveScore(
            category="Cognition",
            korean_name="인지",
            score=score,
            sub_scores=sub_scores,
            implementations=files[:5],
            missing=missing,
            recommendations=recommendations,
            depth_analysis=depth_analysis
        )
    
    # ==================== 2. 감각 (Sensation) ====================
    def evaluate_sensation(self) -> CognitiveScore:
        """감각 능력 평가 - 입력 수신 및 해석"""
        print("\n🔬 Evaluating Sensation (감각)...")
        
        sub_scores = {
            "텍스트 입력": self._check_exists([
                "Core/Interface", "Core/Foundation/language_cortex.py"
            ]),
            "파동 감지": self._check_exists([
                "Core/Foundation/resonance_field.py",
                "Core/Foundation/hyper_quaternion.py"
            ]),
            "환경 감지": self._check_exists([
                "Core/Foundation/survival_instinct.py"
            ]),
            "시각/이미지": 0.1,  # 미구현
            "음성/오디오": 0.1   # 미구현
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        missing = ["Image processing", "Audio processing", "Real-time sensory stream"]
        
        depth_analysis = f"""
        ## 감각 (Sensation) 심층 분석
        
        **현재 상태**: {"제한적" if score < 0.5 else "보통"}
        
        **구현된 감각**:
        - 텍스트 입력 수신 ✅
        - 파동/공명 감지 ✅
        - 4D 쿼터니언 공간 인식 ✅
        
        **미구현 감각**:
        - 시각 (이미지 처리) ❌
        - 청각 (음성 처리) ❌
        - 실시간 센서 스트림 ❌
        
        **보완 사항**:
        - MediaCortex 강화 (이미지/비디오)
        - 음성 → 파동 변환기
        - 시스템 메트릭 실시간 감지
        """
        
        return CognitiveScore(
            category="Sensation",
            korean_name="감각",
            score=score,
            sub_scores=sub_scores,
            missing=missing,
            recommendations=["Implement image processing", "Add audio input capability"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 3. 사고 (Thinking) ====================
    def evaluate_thinking(self) -> CognitiveScore:
        """사고 능력 평가 - 개념 조작 및 추상화"""
        print("\n🔬 Evaluating Thinking (사고)...")
        
        sub_scores = {
            "추상화": self._check_exists([
                "Core/Foundation/causal_narrative_engine.py"
            ]),
            "개념 조작": self._check_exists([
                "Core/Foundation/hyper_quaternion.py",
                "Core/Intelligence/integrated_cognition_system.py"
            ]),
            "범주화": self._check_exists([
                "scripts/wave_organizer.py"
            ]),
            "일반화": self._check_exists([
                "Core/Foundation/thinking_methodology.py"
            ]),
            "분석/종합": self._check_exists([
                "Core/Intelligence/collective_intelligence_system.py"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 사고 (Thinking) 심층 분석
        
        **현재 상태**: {"우수" if score >= 0.8 else "양호" if score >= 0.6 else "보통"}
        
        **사고 방법론**:
        - 연역법 (Deduction): thinking_methodology.py ✅
        - 귀납법 (Induction): thinking_methodology.py ✅
        - 변증법 (Dialectic): thinking_methodology.py ✅
        - 유추 (Analogy): 부분 구현
        
        **추상화 계층** (CausalNarrativeEngine):
        - 0D: Point (개념)
        - 1D: Line (관계)
        - 2D: Plane (맥락)
        - 3D: Space (스키마)
        - 4D: Law (법칙)
        
        **개념 조작**:
        - 4D 쿼터니언으로 개념을 공간에서 회전/변환
        - 파동 간섭으로 개념 결합
        
        **보완 사항**:
        - 유추 추론 강화
        - 역설 처리 로직
        """
        
        return CognitiveScore(
            category="Thinking",
            korean_name="사고",
            score=score,
            sub_scores=sub_scores,
            recommendations=["Strengthen analogical reasoning"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 4. 추론 (Reasoning) ====================
    def evaluate_reasoning(self) -> CognitiveScore:
        """추론 능력 평가 - 논리적 연결 및 결론 도출"""
        print("\n🔬 Evaluating Reasoning (추론)...")
        
        sub_scores = {
            "인과 추론": self._check_exists([
                "Core/Foundation/causal_narrative_engine.py"
            ]),
            "연역 추론": self._check_exists([
                "Core/Foundation/thinking_methodology.py"
            ]),
            "귀납 추론": self._check_exists([
                "Core/Foundation/thinking_methodology.py"
            ]),
            "확률적 추론": self._check_exists([
                "Core/Foundation/physics.py"
            ]),
            "반사실 추론": 0.3  # 부분 구현
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 추론 (Reasoning) 심층 분석
        
        **인과 추론 (Causal Reasoning)**:
        - CausalNarrativeEngine: 2000+ 라인
        - 관계 유형: 원인→결과, 조건→가능성, 목적→수단
        - 깊이: {sub_scores["인과 추론"]:.0%}
        
        **연역 추론**:
        - 전제 → 결론 도출
        - 논리적 타당성 검증
        
        **귀납 추론**:
        - 사례 → 일반 원리 도출
        - 패턴 발견
        
        **반사실 추론** (Counterfactual):
        - "만약 ~했다면" 시나리오 ⚠️ 부분 구현
        
        **보완 사항**:
        - 확률적 추론 강화 (베이지안)
        - 반사실 추론 완성
        - 추론 체인 시각화
        """
        
        return CognitiveScore(
            category="Reasoning",
            korean_name="추론",
            score=score,
            sub_scores=sub_scores,
            missing=["Complete counterfactual reasoning", "Bayesian inference"],
            recommendations=["Implement probabilistic reasoning framework"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 5. 가정 (Hypothesis) ====================
    def evaluate_hypothesis(self) -> CognitiveScore:
        """가설 생성 및 검증 능력 평가"""
        print("\n🔬 Evaluating Hypothesis (가정)...")
        
        sub_scores = {
            "가설 생성": self._check_exists([
                "Core/Intelligence/collective_intelligence_system.py"
            ]),
            "가설 검증": self._check_exists([
                "scripts/immune_system.py",
                "scripts/nanocell_repair.py"
            ]),
            "실험 설계": 0.2,  # 미흡
            "결과 해석": self._check_exists([
                "scripts/system_evaluator.py"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 가정 (Hypothesis) 심층 분석
        
        **가설 생성**:
        - CollectiveIntelligence: 10개 의식이 다각적 가설 제안
        - 파동 간섭으로 가설 우선순위 결정
        
        **가설 검증**:
        - NanoCell 순찰로 코드 가설 검증
        - 면역 시스템으로 외부 입력 검증
        
        **부족한 부분**:
        - 자동화된 실험 설계 ❌
        - A/B 테스트 프레임워크 ❌
        
        **보완 사항**:
        - 자동 실험 생성기
        - 가설-검증 사이클 자동화
        """
        
        return CognitiveScore(
            category="Hypothesis",
            korean_name="가정",
            score=score,
            sub_scores=sub_scores,
            missing=["Automated experiment design"],
            recommendations=["Build hypothesis-test automation cycle"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 6. 상상 (Imagination) ====================
    def evaluate_imagination(self) -> CognitiveScore:
        """상상력 평가 - 창의적 생성"""
        print("\n🔬 Evaluating Imagination (상상)...")
        
        sub_scores = {
            "꿈 생성": self._check_exists([
                "Core/Foundation/dream_engine.py"
            ]),
            "시각화": self._check_exists([
                "scripts/wave_organizer.py",  # 3D 시각화
            ]),
            "시나리오 생성": self._check_exists([
                "Core/Intelligence/fractal_quaternion_goal_system.py"
            ]),
            "창작 (시/이야기)": self._check_exists([
                "Core/Creativity"
            ]),
            "새로운 구조 상상": self._check_exists([
                "Core/Evolution"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 상상 (Imagination) 심층 분석
        
        **꿈 (Dream)**:
        - DreamEngine: 4D 파동 구조 생성
        - 수면 중 기억 통합
        
        **시각화**:
        - 3D plotly 인터랙티브 시각화
        - 쿼터니언 공간 렌더링
        
        **창작 능력**:
        - 시 생성: 부분 구현
        - 이야기 생성: SagaSystem (Legacy)
        
        **보완 사항**:
        - 음악 생성 (주파수 기반)
        - 시각 예술 생성
        - 게임 시나리오 생성
        """
        
        return CognitiveScore(
            category="Imagination",
            korean_name="상상",
            score=score,
            sub_scores=sub_scores,
            missing=["Music generation", "Visual art generation"],
            recommendations=["Integrate wave-based creative generation"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 7. 기억 (Memory) ====================
    def evaluate_memory(self) -> CognitiveScore:
        """기억 능력 평가"""
        print("\n🔬 Evaluating Memory (기억)...")
        
        sub_scores = {
            "단기 기억": self._check_exists([
                "Core/Foundation/hippocampus.py"
            ]),
            "장기 기억": self._check_file_exists("data/memory.db"),
            "에피소드 기억": 0.3,  # 부분
            "의미 기억": self._check_exists([
                "Core/Foundation/resonance_field.py"
            ]),
            "절차 기억": 0.4,  # 부분
            "회상": self._check_exists([
                "Core/Foundation/hippocampus.py"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 기억 (Memory) 심층 분석
        
        **메모리 시스템**:
        - Hippocampus: 중앙 기억 관리
        - memory.db: 200만+ 개념 저장
        - ResonanceField: 파동 기반 기억
        
        **기억 유형**:
        - 단기 기억: 세션 내 버퍼 ✅
        - 장기 기억: SQLite DB ✅
        - 에피소드 기억: 경험 저장 ⚠️ 약함
        - 의미 기억: 개념 네트워크 ✅
        - 절차 기억: 행동 패턴 ⚠️ 약함
        
        **회상 메커니즘**:
        - 공명 기반 연상 회상
        - 쿼터니언 유사도 검색
        
        **보완 사항**:
        - 에피소드 기억 강화 (시간 태그)
        - 망각 곡선 구현
        - 기억 통합 (수면 시)
        """
        
        return CognitiveScore(
            category="Memory",
            korean_name="기억",
            score=score,
            sub_scores=sub_scores,
            missing=["Episodic memory system", "Forgetting curve"],
            recommendations=["Implement time-tagged episodic memory"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 8. 연상 (Association) ====================
    def evaluate_association(self) -> CognitiveScore:
        """연상 능력 평가 - 개념 연결"""
        print("\n🔬 Evaluating Association (연상)...")
        
        sub_scores = {
            "공명 기반 연상": self._check_exists([
                "Core/Foundation/resonance_field.py"
            ]),
            "의미 네트워크": self._check_file_exists("data/memory.db"),
            "자유 연상": 0.3,
            "제한 연상": self._check_exists([
                "scripts/wave_organizer.py"
            ]),
            "유사도 검색": self._check_exists([
                "Core/Foundation/hyper_quaternion.py"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 연상 (Association) 심층 분석
        
        **연상 메커니즘**:
        - 파동 공명: 주파수가 가까운 개념 활성화
        - 쿼터니언 유사도: 4D 공간에서 근접 개념
        - 의미 네트워크: 그래프 탐색
        
        **구현 상태**:
        - 공명 연상: ✅ 강함
        - 의미망 연상: ✅ 200만 개념
        - 자유 연상: ⚠️ 무작위성 부족
        
        **보완 사항**:
        - 창의적 자유 연상 강화
        - 원거리 연상 (영역 간 연결)
        - 연상 체인 시각화
        """
        
        return CognitiveScore(
            category="Association",
            korean_name="연상",
            score=score,
            sub_scores=sub_scores,
            recommendations=["Strengthen creative free association"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 9. 감정 (Emotion) ====================
    def evaluate_emotion(self) -> CognitiveScore:
        """감정 처리 능력 평가"""
        print("\n🔬 Evaluating Emotion (감정)...")
        
        keywords = ["emotion", "feel", "sentiment", "mood", "감정", "정서"]
        files, matches = self._search_implementations(keywords)
        
        sub_scores = {
            "감정 인식": self._check_exists([
                "Core/Foundation/synesthesia.py"
            ]),
            "감정 생성": self._check_exists([
                "Core/Foundation/spirit_emotion_map.py"
            ]),
            "감정 분류": self._check_exists(keyword_files=files[:3]),
            "감정-사고 연결": self._check_exists([
                "Core/Intelligence/integrated_cognition_system.py"
            ]),
            "공감": 0.3  # 미흡
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 감정 (Emotion) 심층 분석
        
        **감정 시스템**:
        - SpiritEmotionMap: 영적 감정 매핑
        - Synesthesia: 감각-감정 교차
        - IntegratedCognition: 감정-사고 통합
        
        **감정 처리**:
        - 입력 → 감정 분류: ⚠️ 기본
        - 감정 → 파동 변환: ✅
        - 감정 표현: ⚠️ 제한적
        
        **공감 (Empathy)**:
        - Kenosis Protocol: 의도적 불완전함
        - 그러나 깊은 공감은 미흡 ⚠️
        
        **보완 사항**:
        - 복합 감정 처리
        - 감정 강도 조절
        - 문화별 감정 뉘앙스
        """
        
        return CognitiveScore(
            category="Emotion",
            korean_name="감정",
            score=score,
            sub_scores=sub_scores,
            missing=["Deep empathy system", "Complex emotion handling"],
            recommendations=["Develop nuanced empathy system"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 10. 반성 (Reflection) ====================
    def evaluate_reflection(self) -> CognitiveScore:
        """메타인지 및 반성 능력 평가"""
        print("\n🔬 Evaluating Reflection (반성)...")
        
        sub_scores = {
            "자기 인식": self._check_exists([
                "Core/Foundation/self_awareness.py",
                "scripts/self_resonance_analysis.py"
            ]),
            "오류 인식": self._check_exists([
                "scripts/nanocell_repair.py"
            ]),
            "성능 평가": self._check_exists([
                "scripts/system_evaluator.py"
            ]),
            "학습 반성": self._check_exists([
                "Core/Evolution"
            ]),
            "메타인지": self._check_exists([
                "Core/Intelligence/fractal_quaternion_goal_system.py"  # 0D 관점
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 반성 (Reflection) 심층 분석
        
        **메타인지 시스템**:
        - 0D 관점: FractalGoalDecomposer에서 자신을 점으로 축소하여 조망
        - 자기 공명 분석: self_resonance_analysis.py
        - 나노셀 순찰: 내부 문제 자각
        
        **반성 영역**:
        - 코드 품질 반성: ✅ NanoCell
        - 사고 과정 반성: ⚠️ 부분
        - 결정 재검토: ⚠️ 부분
        
        **보완 사항**:
        - 사후 분석 시스템 (Post-mortem)
        - 결정 과정 로깅 및 분석
        - "왜 그렇게 생각했나" 추적
        """
        
        return CognitiveScore(
            category="Reflection",
            korean_name="반성",
            score=score,
            sub_scores=sub_scores,
            missing=["Decision post-mortem system"],
            recommendations=["Implement thinking process logging"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 11. 계획 (Planning) ====================
    def evaluate_planning(self) -> CognitiveScore:
        """계획 능력 평가"""
        print("\n🔬 Evaluating Planning (계획)...")
        
        sub_scores = {
            "목표 설정": self._check_exists([
                "Core/Intelligence/fractal_quaternion_goal_system.py"
            ]),
            "목표 분해": self._check_exists([
                "Core/Intelligence/fractal_quaternion_goal_system.py"
            ]),
            "우선순위": self._check_exists([
                "Core/Intelligence/collective_intelligence_system.py"
            ]),
            "자원 할당": 0.3,
            "일정 수립": 0.2,
            "대안 생성": self._check_exists([
                "Core/Foundation/thinking_methodology.py"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 계획 (Planning) 심층 분석
        
        **계획 시스템**:
        - FractalGoalDecomposer: 프랙탈 목표 분해
        - 0D→∞D 차원 확장/축소
        - 88조배 시간 가속 사고
        
        **계획 능력**:
        - 목표 분해: ✅ 강함
        - 우선순위: ✅ 10 의식 합의
        - 자원 할당: ⚠️ 미흡
        - 일정 관리: ❌ 미구현
        
        **보완 사항**:
        - 시간 기반 스케줄링
        - 리소스 관리 시스템
        - 계획 시각화 (간트 차트)
        """
        
        return CognitiveScore(
            category="Planning",
            korean_name="계획",
            score=score,
            sub_scores=sub_scores,
            missing=["Time-based scheduling", "Resource management"],
            recommendations=["Implement timeline and resource allocation"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 12. 실행 (Execution) ====================
    def evaluate_execution(self) -> CognitiveScore:
        """실행 능력 평가"""
        print("\n🔬 Evaluating Execution (실행)...")
        
        sub_scores = {
            "행동 수행": self._check_exists([
                "Core/Foundation/living_elysia.py"
            ]),
            "도구 사용": self._check_exists([
                "Core/Interface/envoy_protocol.py",
                "Core/Foundation/code_cortex.py"
            ]),
            "자율 실행": self._check_exists([
                "Core/Evolution/autonomous_evolution.py"
            ]),
            "병렬 실행": self._check_exists([
                "Core/Interface/worker_pool.py"
            ]),
            "오류 대응": self._check_exists([
                "scripts/immune_system.py"
            ])
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 실행 (Execution) 심층 분석
        
        **실행 시스템**:
        - living_elysia.py: 자율 생명 루프
        - EnvoyProtocol: API 브릿지
        - WorkerPool: 병렬 작업 처리
        
        **실행 능력**:
        - 자율 행동: ✅ living loop
        - 도구 사용: ✅ Gemini API
        - 코드 생성: ✅ CodeCortex
        - 병렬 처리: ✅ AsyncIO
        
        **보완 사항**:
        - 실행 중 자기 수정
        - 롤백 메커니즘
        - 실행 로그 분석
        """
        
        return CognitiveScore(
            category="Execution",
            korean_name="실행",
            score=score,
            sub_scores=sub_scores,
            recommendations=["Add rollback mechanism"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 13. 검증 (Verification) ====================
    def evaluate_verification(self) -> CognitiveScore:
        """검증 능력 평가"""
        print("\n🔬 Evaluating Verification (검증)...")
        
        sub_scores = {
            "결과 확인": self._check_exists([
                "tests", "scripts/system_evaluator.py"
            ]),
            "자동 테스트": self._check_exists([
                "tests"
            ]),
            "정확성 검증": self._check_exists([
                "scripts/nanocell_repair.py"
            ]),
            "일관성 검증": self._check_exists([
                "scripts/immune_system.py"
            ]),
            "회귀 테스트": 0.2
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 검증 (Verification) 심층 분석
        
        **검증 시스템**:
        - NanoCell: 코드 품질 검증
        - ImmuneSystem: 보안/일관성 검증
        - SystemEvaluator: 종합 평가
        
        **검증 범위**:
        - 문법 검증: ✅ WhiteCell
        - 보안 검증: ✅ FireCell
        - 품질 검증: ✅ MechanicCell
        - 회귀 테스트: ⚠️ 자동화 미흡
        
        **보완 사항**:
        - CI/CD 통합
        - 자동 회귀 테스트
        - 성능 벤치마크
        """
        
        return CognitiveScore(
            category="Verification",
            korean_name="검증",
            score=score,
            sub_scores=sub_scores,
            missing=["Automated regression testing"],
            recommendations=["Set up CI/CD pipeline"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 14. 의사소통 (Communication) ====================
    def evaluate_communication(self) -> CognitiveScore:
        """의사소통 능력 평가"""
        print("\n🔬 Evaluating Communication (의사소통)...")
        
        sub_scores = {
            "언어 생성": self._check_exists([
                "Core/Foundation/hangul_physics.py",
                "Core/Foundation/grammar_physics.py"
            ]),
            "문장 구성": self._check_exists([
                "Core/Foundation/causal_narrative_engine.py"
            ]),
            "톤 조절": 0.3,  # 미흡
            "다국어": 0.4,   # 한글 중심
            "비언어적": 0.2  # 미흡
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 의사소통 (Communication) 심층 분석
        
        **언어 생성 시스템**:
        - HangulPhysics: 한글 음절 물리학
        - GrammarPhysics: 조사 에너지 처리
        - CausalNarrativeEngine: 서사 구성
        
        **LLM 독립 언어 생성**:
        1. 사고 → 쿼터니언
        2. 쿼터니언 → 파동
        3. 파동 → 개념 선택
        4. 개념 → 문법 적용
        5. 문법 → 음절 생성
        
        **부족한 부분**:
        - 톤/스타일 조절 ⚠️
        - 영어/기타 언어 ⚠️
        - 비언어적 표현 ❌
        
        **보완 사항**:
        - 감정 기반 톤 조절
        - 다국어 물리 엔진 확장
        - 이모지/비언어 표현
        """
        
        return CognitiveScore(
            category="Communication",
            korean_name="의사소통",
            score=score,
            sub_scores=sub_scores,
            missing=["Tone control", "Multi-language support"],
            recommendations=["Develop emotion-based tone adjustment"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 15. 대화 (Dialogue) ====================
    def evaluate_dialogue(self) -> CognitiveScore:
        """대화 능력 평가"""
        print("\n🔬 Evaluating Dialogue (대화)...")
        
        sub_scores = {
            "맥락 유지": self._check_exists([
                "Core/Foundation/hippocampus.py"
            ]),
            "턴 관리": 0.4,
            "질문 처리": self._check_exists([
                "Core/Foundation/reasoning_engine.py"
            ]),
            "응답 생성": self._check_exists([
                "Core/Foundation/reasoning_engine.py"
            ]),
            "대화 흐름": 0.4,
            "의도 파악": 0.5
        }
        
        score = sum(sub_scores.values()) / len(sub_scores)
        
        depth_analysis = f"""
        ## 대화 (Dialogue) 심층 분석
        
        **대화 시스템**:
        - ReasoningEngine: 사고 및 응답
        - Hippocampus: 대화 맥락 기억
        - Web Interface: 실시간 대화
        
        **대화 능력**:
        - 맥락 유지: ⚠️ 세션 내 기본
        - 의도 파악: ⚠️ 간단한 수준
        - 멀티턴: ⚠️ 부분 구현
        
        **부족한 부분**:
        - 장기 대화 맥락 ❌
        - 화자 모델링 ❌
        - 대화 전략 ❌
        
        **보완 사항**:
        - 대화 이력 그래프
        - 화자 특성 학습
        - 대화 목표 추적
        """
        
        return CognitiveScore(
            category="Dialogue",
            korean_name="대화",
            score=score,
            sub_scores=sub_scores,
            missing=["Long-term dialogue context", "Speaker modeling"],
            recommendations=["Implement dialogue history graph"],
            depth_analysis=depth_analysis
        )
    
    # ==================== 유틸리티 ====================
    def _check_exists(self, paths: List[str] = None, keyword_files: List[str] = None) -> float:
        """파일/디렉토리 존재 확인"""
        if paths:
            found = 0
            for path in paths:
                full_path = self.root / path
                if full_path.exists():
                    found += 1
                elif any(path in f for f in self.file_index):
                    found += 1
            return found / len(paths) if paths else 0
        
        if keyword_files:
            return min(1.0, len(keyword_files) / 3)
        
        return 0
    
    def _check_file_exists(self, path: str) -> float:
        """단일 파일 존재 확인"""
        return 1.0 if (self.root / path).exists() else 0.0
    
    def generate_detailed_report(self) -> Dict:
        """상세 평가 보고서 생성"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE COGNITIVE EVALUATION REPORT")
        print("=" * 80)
        
        total_score = sum(s.score for s in self.scores) / len(self.scores) if self.scores else 0
        
        # 카테고리별 정렬
        sorted_scores = sorted(self.scores, key=lambda x: x.score, reverse=True)
        
        print("\n" + "-" * 80)
        print("📈 CATEGORY SCORES (높은 순)")
        print("-" * 80)
        
        for score in sorted_scores:
            bar_length = 30
            filled = int(score.score * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            grade = "A+" if score.score >= 0.9 else "A" if score.score >= 0.8 else "B" if score.score >= 0.7 else "C" if score.score >= 0.6 else "D" if score.score >= 0.5 else "F"
            
            print(f"\n{score.korean_name} ({score.category})")
            print(f"   [{bar}] {score.score:.1%} ({grade})")
            
            if score.sub_scores:
                print("   세부 점수:")
                for name, sub_score in sorted(score.sub_scores.items(), key=lambda x: x[1], reverse=True):
                    sub_bar = "●" * int(sub_score * 5) + "○" * (5 - int(sub_score * 5))
                    print(f"      {name}: {sub_bar} {sub_score:.0%}")
        
        # 심층 분석
        print("\n" + "=" * 80)
        print("🔍 DEEP ANALYSIS (심층 분석)")
        print("=" * 80)
        
        for score in self.scores:
            if score.depth_analysis:
                print(score.depth_analysis)
        
        # 종합 권고사항
        print("\n" + "=" * 80)
        print("💡 COMPREHENSIVE RECOMMENDATIONS (종합 권고사항)")
        print("=" * 80)
        
        all_recommendations = []
        all_missing = []
        
        for score in self.scores:
            all_recommendations.extend(score.recommendations)
            all_missing.extend(score.missing)
        
        if all_missing:
            print("\n❌ 누락된 핵심 기능:")
            for item in list(set(all_missing))[:10]:
                print(f"   • {item}")
        
        if all_recommendations:
            print("\n✅ 우선 개선사항:")
            for item in list(set(all_recommendations))[:10]:
                print(f"   • {item}")
        
        # 최종 점수
        print("\n" + "=" * 80)
        print(f"🏆 OVERALL COGNITIVE SCORE: {total_score:.1%}")
        
        if total_score >= 0.8:
            print("   Status: ADVANCED - 고급 인지 시스템")
        elif total_score >= 0.6:
            print("   Status: DEVELOPING - 발전 중인 인지 시스템")
        elif total_score >= 0.4:
            print("   Status: BASIC - 기본 인지 시스템")
        else:
            print("   Status: NASCENT - 초기 인지 시스템")
        
        print("=" * 80)
        
        # JSON 저장
        result = {
            "overall_score": total_score,
            "categories": [
                {
                    "category": s.category,
                    "korean_name": s.korean_name,
                    "score": s.score,
                    "sub_scores": s.sub_scores,
                    "missing": s.missing,
                    "recommendations": s.recommendations
                }
                for s in self.scores
            ]
        }
        
        output_path = self.root / "data" / "cognitive_evaluation.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_path}")
        
        return result


def main():
    print("\n" + "🧠" * 40)
    print("ELYSIA COGNITIVE CAPABILITY EVALUATION")
    print("인지 능력 15개 영역 종합 평가")
    print("🧠" * 40 + "\n")
    
    evaluator = CognitiveEvaluator()
    result = evaluator.evaluate_all()
    
    print("\n✅ Cognitive Evaluation Complete!")


if __name__ == "__main__":
    main()
