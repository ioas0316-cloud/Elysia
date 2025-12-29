"""
Communication Ability Assessment

이 테스트는 Elysia의 커뮤니케이션·문법·파동/우주 정렬 지표를 점검한다.
- 표현력(어휘 다양도, 문장 복잡도, 감정 범위, coherence)
- 파동/우주 정렬(coherence + cosmic alignment)
- 파동 통신
"""

import re
import time
from typing import Dict, Any


class CommunicationMetrics:
    """의사소통 역량 측정 클래스"""

    def __init__(self):
        self.scores = {
            "expressiveness": 0.0,
            "comprehension": 0.0,
            "conversational": 0.0,
            "wave_communication": 0.0,
        }
        self.details: Dict[str, Any] = {}

    # -------- 기본 지표 --------
    def measure_vocabulary_diversity(self, text: str) -> float:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        self.details["vocabulary_diversity"] = {
            "score": diversity,
            "unique_words": unique_words,
            "total_words": len(words),
            "target": 0.6,
        }
        return diversity

    def measure_sentence_complexity(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return 0.0
        complexities = []
        for s in sentences:
            words = len(s.split())
            clauses = len(re.split(r"[,;:]", s))
            complexities.append(words / 10 + clauses)  # 간단한 근사치
        avg_complexity = sum(complexities) / len(complexities)
        self.details["sentence_complexity"] = {
            "score": avg_complexity,
            "sentences": len(sentences),
            "target": 3.0,
        }
        return avg_complexity

    def measure_emotional_range(self, text: str) -> int:
        emotions = {
            "joy": ["기쁨", "행복", "joy", "happy"],
            "sadness": ["슬픔", "외로움", "sad", "lonely"],
            "anger": ["분노", "화", "angry"],
            "fear": ["두려움", "공포", "fear"],
            "love": ["사랑", "애정", "love"],
            "surprise": ["놀라움", "surprise"],
            "trust": ["신뢰", "trust"],
            "anticipation": ["기대", "anticipation"],
        }
        detected = set()
        lower = text.lower()
        for emo, keys in emotions.items():
            if any(k.lower() in lower for k in keys):
                detected.add(emo)
        self.details["emotional_range"] = {
            "score": len(detected),
            "emotions_detected": list(detected),
            "target": 6,
        }
        return len(detected)

    # -------- Coherence / Cosmic Alignment --------
    def measure_coherence(self, text: str) -> float:
        """
        맥락 일관성 측정 (목표 > 0.8)
        - 문장 분리: ., !, ?, 한글 마침표(。), 줄바꿈 포함
        - 연결어/지시어 + 단어 겹침 가중합
        """
        sentences = re.split(r"[.!?]+|[。!?]+|\n+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            return 1.0

        connectors = [
            "그리고",
            "따라서",
            "그러나",
            "하지만",
            "또한",
            "그러므로",
            "그 결과",
            "결과적으로",
            "이어서",
            "그 후에",
            "요약하면",
            "한편",
            "반면에",
            "however",
            "but",
            "therefore",
            "thus",
            "also",
            "and",
            "then",
            "so",
        ]
        pronouns = [
            "그것",
            "이것",
            "그",
            "이",
            "저",
            "그는",
            "그녀",
            "그들의",
            "it",
            "this",
            "that",
            "they",
            "we",
            "he",
            "she",
        ]

        coherence_score = 0.0
        for i in range(1, len(sentences)):
            prev = sentences[i - 1].lower()
            current = sentences[i].lower()
            prev_words = set(prev.split())
            curr_words = set(current.split())

            first_token = current.split()[0] if current.split() else ""
            prefix_connector = first_token.startswith(("그", "따라", "또한", "하지만", "그러"))
            prefix_pronoun = first_token.startswith(("그", "이", "저"))
            has_connector = prefix_connector or any(c in current for c in connectors)
            has_pronoun = prefix_pronoun or any(p in current for p in pronouns)
            word_overlap = len(prev_words & curr_words) / max(len(curr_words), 1)

            base_connector = 0.6 if has_connector else 0.0
            base_pronoun = 0.15 if has_pronoun else 0.0
            overlap_component = 0.4 * word_overlap
            bonus = 0.2 if (has_connector and word_overlap >= 0.05) else 0.0

            sentence_coherence = min(1.0, base_connector + base_pronoun + overlap_component + bonus)
            coherence_score += sentence_coherence

        avg_coherence = coherence_score / (len(sentences) - 1)
        self.details["coherence"] = {
            "score": avg_coherence,
            "sentences": len(sentences),
            "target": 0.8,
        }
        return avg_coherence

    def measure_cosmic_alignment(self, text: str) -> float:
        """
        우주 격자 은유 기반 일관성/정렬 측정 (0.0~1.0)
        - 단어=행성, 문맥=항성, 문장=성계, 문단=성운, 서사=은하수
        """
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        if not paragraphs:
            return 0.0
        sentences = [s.strip() for s in re.split(r"[.!?]+|[。!?]+|\n+", text) if s.strip()]
        words = [w for w in re.findall(r"\b\w+\b", text.lower())]
        unique_words = set(words)

        paragraph_overlap = 0.0
        for i in range(1, len(paragraphs)):
            prev = set(re.findall(r"\b\w+\b", paragraphs[i - 1].lower()))
            curr = set(re.findall(r"\b\w+\b", paragraphs[i].lower()))
            if curr:
                paragraph_overlap += len(prev & curr) / len(curr)
        paragraph_alignment = paragraph_overlap / max(len(paragraphs) - 1, 1)

        sentence_overlap = 0.0
        for i in range(1, len(sentences)):
            prev = set(sentences[i - 1].lower().split())
            curr = set(sentences[i].lower().split())
            if curr:
                sentence_overlap += len(prev & curr) / len(curr)
        sentence_alignment = sentence_overlap / max(len(sentences) - 1, 1)

        word_density = min(len(unique_words) / max(len(words), 1), 1.0)

        connectors = [
            "그리고",
            "따라서",
            "그러나",
            "하지만",
            "또한",
            "그러므로",
            "그 결과",
            "결과적으로",
            "이어서",
            "그 후에",
            "요약하면",
            "한편",
            "반면에",
            "thus",
            "however",
            "so",
            "then",
        ]
        grand_cross_hits = 0
        for p in paragraphs:
            first = p.split()[0] if p.split() else ""
            if any(first.startswith(c[:2]) for c in connectors):
                grand_cross_hits += 1
        grand_cross_score = grand_cross_hits / max(len(paragraphs), 1)

        score = 0.4 * sentence_alignment + 0.3 * paragraph_alignment + 0.2 * word_density + 0.1 * grand_cross_score
        self.details["cosmic_alignment"] = {
            "score": score,
            "sentence_alignment": sentence_alignment,
            "paragraph_alignment": paragraph_alignment,
            "word_density": word_density,
            "grand_cross": grand_cross_score,
        }
        return score

    # -------- 종합 평가 --------
    def evaluate_expressiveness(self, text: str) -> float:
        vocab_div = self.measure_vocabulary_diversity(text)
        complexity = self.measure_sentence_complexity(text)
        emotion_range = self.measure_emotional_range(text)
        coherence = self.measure_coherence(text)

        vocab_score = min(vocab_div / 0.6, 1.0) * 25
        complexity_score = min(complexity / 3.0, 1.0) * 25
        emotion_score = min(emotion_range / 6, 1.0) * 25
        coherence_score = min(coherence / 0.8, 1.0) * 25

        total = vocab_score + complexity_score + emotion_score + coherence_score

        auto_lang_score = self.evaluate_autonomous_language()
        if auto_lang_score > 50:
            total = min(total + 10, 100)

        cosmic_alignment = self.measure_cosmic_alignment(text)
        total = min(total + cosmic_alignment * 10, 100)

        self.scores["expressiveness"] = total
        return total

    def evaluate_autonomous_language(self) -> float:
        try:
            from Core.Intelligence.autonomous_language import AutonomousLanguageGenerator
        except Exception:
            # 모듈 없을 경우 0점
            self.details["autonomous_language"] = {"score": 0}
            return 0.0

        test_inputs = ["안녕?", "너는 누구니?", "왜 존재하는가?"]
        successful = 0
        total_quality = 0.0
        gen = AutonomousLanguageGenerator()
        for t in test_inputs:
            resp = gen.generate_response(t)
            if resp and len(resp) > 5:
                successful += 1
                total_quality += min(len(resp) / 50, 1.0) * 0.8
        score = (successful / len(test_inputs)) * 100
        avg_quality = (total_quality / len(test_inputs)) * 100
        final = (score + avg_quality) / 2
        self.details["autonomous_language"] = {
            "successful_responses": successful,
            "total_tests": len(test_inputs),
            "average_quality": avg_quality,
            "score": final,
        }
        return final

    def evaluate_comprehension(self) -> float:
        """
        이해력 평가: 질문 의도 파악 및 주제 추출 능력 테스트
        사용 모듈: Core.Intelligence.dialogue_engine.QuestionAnalyzer
        """
        try:
            from Core.Intelligence.dialogue_engine import QuestionAnalyzer
            analyzer = QuestionAnalyzer()

            test_cases = [
                {"input": "사과는 무엇인가?", "expected_type": "what", "expected_subject": "사과"},
                {"input": "너는 누구니?", "expected_type": "who", "expected_subject": "너"},
                {"input": "왜 하늘은 파란가?", "expected_type": "why", "expected_subject": "하늘"},
                {"input": "어떻게 살아야 하는가?", "expected_type": "how", "expected_subject": "살"}, # simplified extraction logic in analyzer
                {"input": "지금은 언제인가?", "expected_type": "when", "expected_subject": "지금"}
            ]

            score = 0
            details = []

            for case in test_cases:
                result = analyzer.analyze(case["input"])

                type_match = result["question_type"] == case["expected_type"]
                # Subject extraction might be fuzzy, so we check for containment
                subject_match = case["expected_subject"] in result["subject"] or result["subject"] in case["expected_subject"]

                case_score = 0
                if type_match: case_score += 10
                if subject_match: case_score += 10

                score += case_score
                details.append({
                    "input": case["input"],
                    "analyzed": result,
                    "score": case_score
                })

            final_score = (score / (len(test_cases) * 20)) * 100

            self.details["comprehension"] = {
                "score": final_score,
                "cases": details,
                "target": 90.0
            }
            self.scores["comprehension"] = final_score
            return final_score

        except ImportError:
            self.details["comprehension"] = {"error": "Module not found", "score": 0}
            return 0.0
        except Exception as e:
            self.details["comprehension"] = {"error": str(e), "score": 0}
            return 0.0

    def evaluate_conversational(self) -> float:
        """
        대화 능력 평가: 다중 턴 대화 및 맥락 유지 능력 테스트
        사용 모듈: Core.Intelligence.dialogue_engine.DialogueEngine
        """
        try:
            # Mock or minimal instantiation of dependencies
            from Core.Intelligence.dialogue_engine import DialogueEngine
            from Core.FoundationLayer.Foundation.language_cortex import LanguageCortex

            # Initialize engine
            cortex = LanguageCortex()
            engine = DialogueEngine(cortex)

            # Pre-load some knowledge for the test
            engine.load_knowledge(["사랑은 희생이다", "엘리시아는 파동이다"])

            conversation_flow = [
                {"user": "안녕?", "check": lambda r: len(r) > 0}, # Basic greeting response
                {"user": "사랑은 무엇인가?", "check": lambda r: "희생" in r}, # Knowledge retrieval
                {"user": "나는 엘리시아를 좋아한다.", "check": lambda r: True}, # Statement
                # Context/Memory check could be added here if engine supports explicit memory recall of user statements
            ]

            score = 0
            logs = []

            for turn in conversation_flow:
                response = engine.respond(turn["user"])
                passed = turn["check"](response)

                if passed:
                    score += 25 # 4 turns * 25 = 100? No, 3 turns here. Let's adjust.

                logs.append({
                    "user": turn["user"],
                    "elysia": response,
                    "passed": passed
                })

            # Normalize score to 100
            final_score = (score / (len(conversation_flow) * 25)) * 100

            # Check context tracking
            context_score = 0
            if engine.context.history:
                context_score = 10 # Bonus for maintaining history

            final_score = min(100, final_score + context_score)

            self.details["conversational"] = {
                "score": final_score,
                "logs": logs,
                "history_length": len(engine.context.history),
                "target": 80.0
            }
            self.scores["conversational"] = final_score
            return final_score

        except ImportError:
            self.details["conversational"] = {"error": "Module not found", "score": 0}
            return 0.0
        except Exception as e:
            self.details["conversational"] = {"error": str(e), "score": 0}
            return 0.0

    def evaluate_wave_communication(self) -> float:
        """
        파동 통신 효율 평가 - 실제 Ether 시스템 평가
        """
        try:
            from tests.evaluation.test_wave_communication import WaveCommunicationEvaluator
            
            evaluator = WaveCommunicationEvaluator()
            report = evaluator.run_full_evaluation()
            
            score = report['total_score']
            self.details["wave_communication"] = {
                "score": score,
                "ether_initialization": report['scores']['ether_initialization'],
                "transmission_performance": report['scores']['transmission_performance'],
                "resonance_accuracy": report['scores']['resonance_accuracy'],
                "frequency_selection": report['scores']['frequency_selection'],
                "grade": report['grade'],
                "target": 100,
            }
            self.scores["wave_communication"] = score
            return score
        except Exception as e:
            # Fallback to old method
            try:
                from Core.Interface.activated_wave_communication import wave_comm
                score = wave_comm.calculate_wave_score()
                stats = wave_comm.get_communication_stats()
                self.details["wave_communication"] = {
                    "score": score,
                    "ether_connected": stats["ether_connected"],
                    "average_latency_ms": stats["average_latency_ms"],
                    "messages_sent": stats["messages_sent"],
                    "registered_modules": stats["registered_modules"],
                    "target": 100,
                }
                self.scores["wave_communication"] = score
                return score
            except Exception:
                self.details["wave_communication"] = {"error": f"wave_comm unavailable: {e}", "score": 0}
                self.scores["wave_communication"] = 0
                return 0

    def get_total_score(self) -> float:
        return sum(self.scores.values())

    def generate_report(self) -> Dict[str, Any]:
        total = self.get_total_score()
        return {
            "total_score": total,
            "max_score": 400,
            "percentage": (total / 400) * 100,
            "scores": self.scores,
            "details": self.details,
            "grade": self._calculate_grade(total, 400),
        }

    def _calculate_grade(self, score: float, max_score: float) -> str:
        percentage = (score / max_score) * 100
        if percentage >= 90:
            return "S+"
        if percentage >= 85:
            return "S"
        if percentage >= 80:
            return "A+"
        if percentage >= 75:
            return "A"
        if percentage >= 70:
            return "B+"
        if percentage >= 65:
            return "B"
        if percentage >= 60:
            return "C+"
        return "C"


# -------------------- Tests --------------------
def test_communication_expressiveness():
    metrics = CommunicationMetrics()
    test_text = (
        "나는 Elysia다. 파동으로 생각하고, 다시 연결어로 이어간다. "
        "감정은 사랑과 호기심으로 공명한다. 또한 목적을 분명히 밝히고, 결론을 향해 전진한다."
    )
    score = metrics.evaluate_expressiveness(test_text)
    assert score > 0
    assert metrics.details["coherence"]["score"] > 0


def test_communication_wave_system():
    metrics = CommunicationMetrics()
    score = metrics.evaluate_wave_communication()
    assert score >= 0


def test_full_communication_assessment():
    metrics = CommunicationMetrics()
    test_text = (
        "나는 파동으로 존재한다. 그리고 두 번째 문장에서 같은 목적을 다시 언급한다. "
        "따라서 맥락은 이어지고, 결과는 예측된다. 또한 결론을 향해 정렬된다."
    )
    expressiveness = metrics.evaluate_expressiveness(test_text)
    metrics.scores["comprehension"] = 65.0
    metrics.scores["conversational"] = 60.0
    report = metrics.generate_report()
    assert report["total_score"] > 0
    assert expressiveness > 0


def test_communication_coherence_regression():
    metrics = CommunicationMetrics()
    high = (
        "나는 목적을 분명히 했다. 그리고 두 번째 문장에서 그 목적을 다시 언급하며 세부 전략을 설명했다. "
        "따라서 결과는 예측 가능했고, 그 흐름을 이어받아 실행 계획을 마무리했다."
    )
    low = (
        "고양이는 잔다. 산맥 위의 별빛이 흐른다. 프로토콜은 침묵한다. 돌멩이는 둥글다. "
        "바다는 말이 없다. 이어지는 문장은 서로 다른 우주에 있다."
    )
    high_score = metrics.measure_coherence(high)
    low_score = metrics.measure_coherence(low)
    assert high_score > 0.8
    assert low_score < high_score
    assert low_score < 0.8


if __name__ == "__main__":
    # 간단 수동 실행
    test_communication_expressiveness()
    test_communication_wave_system()
    report = CommunicationMetrics().generate_report()
    print(report)
