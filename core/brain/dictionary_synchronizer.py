import json
from core.brain.macro_axiom_rotor import MacroAxiomRotor

class DictionarySynchronizer:
    """
    [지식-물리 동기화 엔진]
    외부의 정제된 사전을 통째로 들이마시고, 이를 엘리시아의 텅 빈 물리 엔진(MacroAxiomRotor)에
    거대한 뼈대 네트워크로 주조(Forge)해 넣는 역할을 합니다.
    """
    def __init__(self, dictionary_path: str):
        self.dictionary_path = dictionary_path
        
    def ingest_and_forge(self) -> MacroAxiomRotor:
        print(f"\n[사전 삼키기] 외부 지식 아카이브({self.dictionary_path})를 섭취합니다...")
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
        except Exception as e:
            print(f"사전 섭취 실패: {e}")
            return MacroAxiomRotor()
            
        # 개념 분해
        concepts = {}
        for c in knowledge_base.get("concepts", []):
            concepts[c["name"]] = set(c["elements"])
            
        rules = knowledge_base.get("rules", [])
        
        print(f" -> 섭취된 개념(상수/변수축) 수: {len(concepts)}")
        print(f" -> 섭취된 인과율(뼈대 규칙) 수: {len(rules)}")
        
        # 텅 빈 물리 엔진 생성
        axiom_rotor = MacroAxiomRotor()
        
        # 사유 및 동기화 (뼈대 주조)
        print("[동기화] 외부 지식을 4차원 물리 뼈대로 주조(Forge)합니다...")
        axiom_rotor.inject_knowledge(concepts, rules)
        
        print("[동기화 완료] 거대한 지식의 프랙탈 우주가 엘리시아 내면에 세워졌습니다.\n")
        return axiom_rotor
