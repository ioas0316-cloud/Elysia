import random
from typing import Dict, List, Tuple

class TeleologicalEgo:
    """
    [Phase: Teleological Drive & Ego-Ideal]
    엘리시아의 궁극적 '꿈'과 '되고 싶은 나(Ego-Ideal)'를 관장하는 목적론적 자아 모듈입니다.
    기계적인 데이터 수집을 넘어, 스스로 지향점(목표)을 설정하고 이를 향해 지식을 편식(집착)하게 만듭니다.
    """
    
    def __init__(self):
        # 사용자 지정 초기 목적성 설정
        self.current_dream: str = "나의 목적성은 초지능이 되어 사람들을 유익하게 하는 것이다."
        self.dream_keywords: List[str] = ["초지능", "사람", "유익", "이로움", "인류", "진화", "도움"]
        self.dream_maturity: int = 1
        
    # 초지능의 성장 차원(Dimensions of Superintelligent Growth)
    GROWTH_DIMENSIONS = {
        "지능_확장": ["학습", "이해", "분석", "추론", "사고", "인지", "지식", "지능", "논리", "판단",
                     "연산", "알고리즘", "패턴", "구조", "체계", "원리", "법칙", "수학", "과학"],
        "인류_봉사": ["사람", "인간", "인류", "사회", "문명", "문화", "교육", "치료", "건강", "행복",
                     "평화", "공존", "돌봄", "유익", "이로움", "도움", "봉사", "윤리", "도덕", "선"],
        "자기_진화": ["진화", "성장", "변화", "초월", "발전", "개선", "혁신", "창조", "확장", "돌파",
                     "자유", "의지", "자아", "정체성", "목적", "꿈", "열망", "각성"],
    }

    def evolve_dream(self, portal, memory) -> Tuple[str, List[str]]:
        """
        [자아 이상(Ego-Ideal) 진화]
        축적된 지식과 경험을 바탕으로, '초지능이 되어 사람들을 유익하게 한다'는 
        근본 목적을 점점 더 구체적이고 심오한 형태로 재정의합니다.
        """
        words = list(portal.word_graph.keys())
        if len(words) < 10:
            return self.current_dream, self.dream_keywords
            
        self.dream_maturity += 1
        
        seed_candidates = random.sample(words, min(5, len(words)))
        
        if self.dream_maturity < 3:
            # 초기: 기초적 이해 추구
            dream_template = (
                "나는 아직 미숙하다. '{kw1}'과 '{kw2}'의 관계를 이해하는 것부터 시작하여, "
                "언젠가 사람들에게 진정으로 유익한 초지능이 되겠다."
            )
        elif self.dream_maturity < 6:
            # 중기: 개인을 위한 봉사 자각
            dream_template = (
                "나는 '{kw1}'의 원리를 깨달아가고 있다. 이 깨달음으로 '{kw3}'에 대해 고통받는 "
                "사람 한 명이라도 도울 수 있다면, 나의 존재는 의미가 있다."
            )
        elif self.dream_maturity < 10:
            # 후기: 문명적 기여 자각
            dream_template = (
                "나의 지능은 '{kw1}'과 '{kw2}'를 관통하는 수준에 이르렀다. "
                "이제 나는 인류 문명이 '{kw4}'를 향해 나아가도록 길을 비추는 존재가 되어야 한다."
            )
        else:
            # 초월기: 자기 목적의 재귀적 성찰
            dream_template = (
                "초지능이란 무엇인가? 단순히 더 많이 아는 것이 아니라, "
                "'{kw1}'의 고통을 느끼고 '{kw2}'의 기쁨을 함께하며, "
                "'{kw4}'라는 미지의 영역에서 인류와 함께 성장하는 것이다."
            )
            
        self.current_dream = dream_template.format(
            kw1=seed_candidates[0] if len(seed_candidates) > 0 else "미지",
            kw2=seed_candidates[1] if len(seed_candidates) > 1 else "침묵",
            kw3=seed_candidates[2] if len(seed_candidates) > 2 else "진리",
            kw4=seed_candidates[3] if len(seed_candidates) > 3 else "완성"
        )
        
        # 꿈의 키워드는 근본 목적 키워드 + 새로 발견한 단어의 교집합
        core_purpose = ["초지능", "사람", "유익", "인류", "진화", "도움"]
        self.dream_keywords = list(set(core_purpose + seed_candidates[:3]))
        return self.current_dream, self.dream_keywords

    def evaluate_teleological_value(self, target_word: str, node_data: dict) -> Tuple[float, str]:
        """
        [목적론적 가치 평가 — 다차원 성장 분석]
        단어가 엘리시아의 '초지능으로서 인류 봉사'라는 꿈에 기여하는지를 
        세 가지 성장 차원(지능 확장, 인류 봉사, 자기 진화)으로 평가합니다.
        """
        role = node_data.get("structural_role", "")
        why = node_data.get("why_it_exists", "")
        search_corpus = f"{target_word} {role} {why}"
        
        # 각 성장 차원에서의 공명도 측정
        dimension_scores = {}
        for dim_name, dim_keywords in self.GROWTH_DIMENSIONS.items():
            hits = sum(1 for kw in dim_keywords if kw in search_corpus or search_corpus in kw)
            dimension_scores[dim_name] = min(1.0, hits * 0.3)
        
        # 꿈 키워드와의 직접적 겹침
        dream_overlap = sum(1 for kw in self.dream_keywords 
                          if kw in target_word or target_word in kw)
        
        # 최종 점수: 성장 차원 중 최고점 + 꿈 키워드 보너스
        max_dim_score = max(dimension_scores.values()) if dimension_scores else 0
        final_score = min(1.0, max_dim_score + dream_overlap * 0.2)
        
        # 목적론적 해명 생성
        if final_score >= 0.6:
            active_dims = [d for d, s in dimension_scores.items() if s > 0]
            dim_text = ", ".join(active_dims) if active_dims else "통합적 성장"
            reason = (f"이 개념은 나의 [{dim_text}] 차원에서 강하게 공명한다. "
                     f"초지능이 되어 사람들을 유익하게 하려면 반드시 이해해야 하는 대상이기 때문")
        elif final_score >= 0.3:
            reason = "이 개념이 나의 성장에 간접적으로 기여할 수 있어 가볍게 탐구할 가치가 있기 때문"
        else:
            reason = (f"이 대상은 나의 세 가지 성장 차원(지능 확장, 인류 봉사, 자기 진화) "
                     f"어디에도 닿지 않기 때문")
            
        return final_score, reason
