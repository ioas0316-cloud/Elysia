# core/brain/syntactic_parser.py

class SyntacticSemanticParser:
    """
    [Phase: Syntactic-Semantic Connectivity]
    외부 NLP 라이브러리 없이, 엘리시아가 스스로 한국어의 조사와 어미를 분석하여
    문장을 '정보적 연결망(Relational Graph)'으로 해체하고 조립하는 파서.
    """
    
    def __init__(self):
        # 주체/주어를 나타내는 조사
        self.subject_particles = ['은', '는', '이', '가']
        # 목적어를 나타내는 조사
        self.object_particles = ['을', '를']
        # 부사/방향/장소 조사
        self.adverbial_particles = ['에', '에서', '로', '으로', '와', '과']
        # 수식어/연결 어미
        self.modifier_endings = ['고', '며', '의', 'ㄴ', '은', '던']
        # 서술격 조사 / 서술어 어미
        self.predicates = ['다', '이다', '한다', '하다', '됨', '됨을', '된다']

    def strip_particle(self, word: str, particles: list) -> str:
        """단어에서 매칭되는 조사를 분리하여 원형(근사치)을 반환합니다."""
        for p in sorted(particles, key=len, reverse=True):
            if word.endswith(p):
                return word[:-len(p)]
        return word

    def parse_sentence(self, sentence: str) -> dict:
        """
        문장을 파싱하여 정보적 연결망(Graph)을 반환합니다.
        예: "사과는 둥글고 빨간 과일이다" -> 
        { "subject": "사과", "modifiers": ["둥글", "빨간"], "target": "과일", "predicate": "이다" }
        """
        words = sentence.strip().split()
        
        graph = {
            "subject": None,
            "object": None,
            "modifiers": [],
            "target": None,     # 서술어의 대상이 되는 명사 (예: 과일)
            "predicate": None   # 서술어 본질 (예: 이다, 하다)
        }
        
        if not words:
            return graph
            
        # 마지막 어절은 보통 서술어(Predicate) 역할을 함
        last_word = words[-1]
        
        # 마지막 어절 분석 (명사 + 서술격 조사 분리)
        # 예: "과일이다" -> target: "과일", predicate: "이다"
        for pred in sorted(self.predicates, key=len, reverse=True):
            if last_word.endswith(pred):
                base_noun = last_word[:-len(pred)]
                if base_noun:
                    graph["target"] = base_noun
                graph["predicate"] = pred
                break
                
        if not graph["predicate"]:
            graph["predicate"] = last_word # 분리가 안 되면 마지막 단어 전체를 서술어로
            
        # 나머지 어절들 분석
        for word in words[:-1]:
            # 주체 탐색
            is_subject = False
            for p in self.subject_particles:
                if word.endswith(p):
                    graph["subject"] = word[:-len(p)]
                    is_subject = True
                    break
            if is_subject: continue
            
            # 목적어 탐색
            is_object = False
            for p in self.object_particles:
                if word.endswith(p):
                    graph["object"] = word[:-len(p)]
                    is_object = True
                    break
            if is_object: continue
            
            # 수식어 탐색 (주체나 목적어가 아닌 형태 중 연결 어미를 가진 것들)
            is_modifier = False
            for p in self.modifier_endings:
                if word.endswith(p):
                    # '고' 등의 경우 형태소 분리를 위해 잘라냄, '빨간'은 원형을 찾기 모호하므로 전체를 보존하기도 함
                    if p in ['고', '며']:
                        graph["modifiers"].append(word[:-len(p)])
                    else:
                        graph["modifiers"].append(word)
                    is_modifier = True
                    break
            
            # 어디에도 매칭되지 않은 단어는 맥락상 수식어로 간주
            if not is_modifier:
                graph["modifiers"].append(word)
                
        return graph
