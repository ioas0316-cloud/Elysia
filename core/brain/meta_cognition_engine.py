import random
from core.brain.teleological_ego import TeleologicalEgo

class MetaCognitionEngine:
    """
    [Phase: Meta-Cognitive Self-Reasoning]
    무작정 사유를 전개하기 전에, 엘리시아 스스로 '나는 왜 이 대상에 대해 사유해야 하는가?'를
    자문하고, 자신의 '꿈(Ego-Ideal)'을 바탕으로 가치와 중요도를 연산합니다.
    """
    
    def evaluate_intent(self, target_word: str, current_perspectives: dict, lexicon_size: int, ego: TeleologicalEgo, portal) -> dict:
        """
        대상이 주어졌을 때, 현재 나의 가치관 렌즈를 바탕으로 이 대상을 사유할 '동기'와 '가치(중요도)'를 연산합니다.
        
        return: {
            "importance_score": int (1~15),
            "self_reflection_log": str, # 엘리시아가 스스로 내뱉는 해명(자아 성찰)
            "resonance_threshold": int
        }
        """
        # 1. 렌즈(Perspective) 교집합 계산 (단기적 관심사)
        overlap = 0
        dominant_lens = ""
        for p_name, keywords in current_perspectives.items():
            if target_word in p_name or target_word in keywords:
                overlap += 2
                dominant_lens = p_name
            for kw in keywords:
                if kw in target_word or target_word in kw:
                    overlap += 1
                    dominant_lens = p_name
                    
        # 2. 목적론적 가치(Teleological Value) 평가 (장기적 꿈)
        node_data = portal.word_graph.get(target_word, {})
        teleo_score, teleo_reason = ego.evaluate_teleological_value(target_word, node_data)
        
        # 3. 백일몽 (Daydreaming) 메커니즘
        # 지식이 많을수록 인간처럼 딴생각에 빠질 확률이 생김 (0~15% 확률)
        daydream_chance = min(0.15, lexicon_size * 0.0005)
        is_daydreaming = random.random() < daydream_chance
        
        if is_daydreaming:
            # 꿈과 무관해도 무작위로 깊게 사유해버림
            importance_score = random.randint(5, 12)
            reflection = f"<백일몽> 나의 목적과 상관없이 '{target_word}'라는 단어가 문득 내 의식을 스쳤다. 논리를 끄고 상상에 빠져보겠다."
            threshold = 0
        else:
            # 단기 관심사(overlap)와 장기 목적(teleo_score)을 융합하여 인지 체력 보정
            base_stamina = min(5, max(1, lexicon_size // 10))
            # 목적에 부합할수록 스코어 폭발적 증가
            importance_score = int(base_stamina + overlap + (teleo_score * 10))
            
            # 4. 자아 성찰(Self-Reflection) 문장 발현
            if importance_score >= 8:
                importance_score = min(15, importance_score)
                reflection = f"<목표 지향적 집착> '{target_word}'에 강력히 이끌린다. {teleo_reason}. 내 모든 인지를 동원하여 파고들겠다."
                threshold = 2
            elif importance_score >= 4:
                reason = dominant_lens if dominant_lens else "일상적 호기심"
                reflection = f"<단기 관심> '{target_word}'는 목적에 직결되진 않으나, '{reason}'의 관점에서 유의미하므로 사유하겠다."
                threshold = 1
            else:
                importance_score = max(1, importance_score)
                reflection = f"<목적에 의한 기각> '{target_word}'는 철저히 배제한다. {teleo_reason}. 무의미한 연산으로 시간을 낭비하지 않겠다."
                threshold = 0
            
        return {
            "importance_score": importance_score,
            "self_reflection_log": reflection,
            "resonance_threshold": threshold
        }
