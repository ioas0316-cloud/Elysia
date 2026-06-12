import os
import json
import math

class LanguagePortalEngine:
    """
    [Phase 151] 진정한 무개입 사전 관측소 (Zero-Intervention Lexicon Topology)
    
    기존의 임의적 한글 초성 축(Axis)이나 해시 기반 각도 변환을 완전히 제거했습니다.
    이 엔진은 단순히 10만 단어의 '단어 -> 정의에 포함된 단어들' 이라는
    원시적인 거대 연결망(Graph) 그 자체만을 보존합니다.
    구조 원리는 이 지형을 걸어가는 과정에서 스스로 발견되어야 합니다.
    """
    def __init__(self, lexicon_path=None):
        if lexicon_path is None:
            self.lexicon_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "deep_korean_lexicon.json")
        else:
            self.lexicon_path = lexicon_path
            
        self.lexicon = {}
        self.word_graph = {}  # 단어 -> {정의 및 구조적 연결 정보}
        
        self._load_and_map()

    def _load_and_map(self):
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            self.lexicon = json.load(f)
            
        all_words = set(self.lexicon.keys())
        
        for word, data in self.lexicon.items():
            if isinstance(data, dict):
                # Deep Korean Lexicon Format
                structural_role = data.get("structural_role", "")
                why_it_exists = data.get("why_it_exists", "")
                connections = data.get("connections", {})
                
                binds_to = connections.get("binds_to", "")
                if isinstance(binds_to, str):
                    bind_list = [binds_to]
                else:
                    bind_list = binds_to
                    
                # Extract any recognized words from its values
                all_text = f"{structural_role} {why_it_exists} {' '.join(bind_list)}"
                connected_nodes = set()
                # Simple extraction of other dictionary words found in the text
                for dw in all_text.replace(",", " ").replace("(", " ").replace(")", " ").split():
                    clean_dw = "".join(c for c in dw if c.isalnum()).lower()
                    if clean_dw in all_words and clean_dw != word:
                        connected_nodes.add(clean_dw)
                
                self.word_graph[word] = {
                    "definition": why_it_exists,
                    "structural_role": structural_role,
                    "syntactic_trajectory": connections.get("syntactic_trajectory", ""),
                    "connections": list(connected_nodes)
                }
            else:
                # Legacy WordNet fallback
                definition = data
                words_in_def = definition.split()
                connected_nodes = set()
                for dw in words_in_def:
                    clean_dw = "".join(c for c in dw if c.isalnum()).lower()
                    if clean_dw in all_words and clean_dw != word:
                        connected_nodes.add(clean_dw)
                self.word_graph[word] = {
                    "definition": definition,
                    "structural_role": "Unknown",
                    "syntactic_trajectory": "",
                    "connections": list(connected_nodes)
                }

    def add_concept(self, word: str, structural_role: str, why_it_exists: str, binds_to: list, syntactic_trajectory: str) -> bool:
        """
        [Phase: Self-Expanding Lexicon]
        새롭게 깨달은 이치(개념)를 사전에 스스로 편입합니다.
        """
        if word in self.lexicon:
            return False
            
        self.lexicon[word] = {
            "structural_role": structural_role,
            "why_it_exists": why_it_exists,
            "connections": {
                "binds_to": binds_to,
                "syntactic_trajectory": syntactic_trajectory
            }
        }
        
        try:
            with open(self.lexicon_path, 'w', encoding='utf-8') as f:
                json.dump(self.lexicon, f, ensure_ascii=False, indent=2)
            self._load_and_map()
            return True
        except Exception:
            return False
