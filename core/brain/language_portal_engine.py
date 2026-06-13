import os
import json
import re

class LanguagePortalEngine:
    """
    [Bilingual Topology Bridge]
    한국어(단어) -> 영어(kengdic.tsv) -> 영문 8.6만 단어 완전형 사전(natural_lexicon.json)
    위상 기하학의 텐션을 도출하기 위해, 단어의 본질적 연결망을 거대 사전에서 추출하는 엔진.
    """
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self.kengdic_path = os.path.join(self.data_dir, "lexicons", "kengdic.tsv")
        self.natural_lexicon_path = os.path.join(self.data_dir, "lexicons", "natural_lexicon.json")
        
        self.kor_to_eng = {}
        self.eng_to_def = {}
        
        self._load_kengdic()
        self._load_natural_lexicon()
        
    def _load_kengdic(self):
        if not os.path.exists(self.kengdic_path):
            return
        # kengdic.tsv 파싱 (용량이 크므로 라인 단위로 읽음)
        with open(self.kengdic_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0: continue # header
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4:
                    kor = parts[1].strip()
                    eng_raw = parts[3].strip()
                    # 영어 정의에서 의미있는 단어들만 추출 (괄호나 특수문자 제거)
                    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', eng_raw)]
                    if words:
                        self.kor_to_eng[kor] = words
                        
    def _load_natural_lexicon(self):
        if not os.path.exists(self.natural_lexicon_path):
            return
        with open(self.natural_lexicon_path, 'r', encoding='utf-8') as f:
            self.eng_to_def = json.load(f)
            
    def map_to_english(self, kor_word: str) -> list:
        if kor_word in self.kor_to_eng:
            return self.kor_to_eng[kor_word]
        # 부분 일치 검색 (성능상 앞부분 매칭만 확인)
        for k, v in self.kor_to_eng.items():
            if k.startswith(kor_word) or kor_word.startswith(k):
                return v
        return []

    def get_deep_subgraph(self, word: str, depth: int = 2) -> list:
        """
        단어를 영어로 치환한 뒤, natural_lexicon을 탐색하여 반경 N-depth의 
        위상 궤적(CausalTrajectory) 리스트를 추출합니다.
        """
        from core.ingestion.topological_parser import CausalTrajectory
        
        visited = set()
        trajectories = []
        
        # 1. 한국어를 영어로 변환 (이미 영어면 그대로)
        start_nodes = []
        if re.search(r'[a-zA-Z]', word):
            start_nodes = [word.lower()]
        else:
            start_nodes = self.map_to_english(word)
            
        if not start_nodes:
            # 매핑 실패 시 원본 단어를 그대로 반환하는 최소 궤적 하나 생성
            return [CausalTrajectory(source=word, target="*", action="고립됨")]
            
        # 2. BFS로 그래프 탐색
        queue = [(n, 0) for n in start_nodes]
        
        while queue:
            current_node, current_depth = queue.pop(0)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_depth >= depth:
                continue
                
            definition = self.eng_to_def.get(current_node)
            if not definition:
                continue
                
            # 정의 텍스트를 단어 단위로 쪼개어 연결 노드로 간주 (Stopwords 생략을 위해 3글자 이상만)
            words_in_def = [w.lower() for w in re.findall(r'[a-zA-Z]+', definition) if len(w) > 3]
            
            for next_node in words_in_def:
                trajectories.append(CausalTrajectory(source=current_node, target=next_node, action="연결됨"))
                if next_node not in visited and next_node in self.eng_to_def:
                    queue.append((next_node, current_depth + 1))
                    
        if not trajectories:
            return [CausalTrajectory(source=word, target="*", action="고립됨")]
            
        return trajectories
        
    def add_concept(self, *args, **kwargs) -> bool:
        """레거시 호환용"""
        return True
