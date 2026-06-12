import os

class LinguisticRotor:
    """
    [Phase: Pure Semantic Layer]
    기하학, 좌표계, 수학을 전면 배제합니다.
    사전(Lexicon)에 적힌 단어의 속성(의미, 존재 이유, 역할)만을 바탕으로,
    단어와 단어가 어떻게 이어지는지 공통 원리를 찾아가는 순수 사유 레이어입니다.
    """
    
    def __init__(self, lexicon_path: str = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        if lexicon_path is None:
            self.lexicon_path = os.path.join(self.base_dir, "..", "..", "data", "deep_korean_lexicon.json")
        else:
            self.lexicon_path = lexicon_path
            
        try:
            from core.brain.language_portal_engine import LanguagePortalEngine
            self.portal = LanguagePortalEngine(self.lexicon_path)
            self.words = list(self.portal.word_graph.keys())
        except:
            self.portal = None
            self.words = []

    def achieve_semantic_consensus(self, word_x: str, word_y: str, word_z: str, perspectives: dict = None, importance_score: int = 3, resonance_threshold: int = 0) -> list:
        """
        [Phase: Value-Driven Semantic Expansion]
        엘리시아 스스로 판단한 중요도(importance_score)에 비례하여 사유의 깊이(스텝)가 스케일링됩니다.
        중요하지 않으면 1단계만 보고 멈추지만, 중요하면 심연까지 추적합니다.
        
        return: list of {"equilibrium_word": str, "trajectory": list[str], "perspective": str}
        """
        if not self.portal or not self.words:
            return [{"equilibrium_word": "침묵", "trajectory": [word_x, "침묵"], "perspective": "None"}]

        if not perspectives:
            perspectives = {"원초적 관점": ["존재", "탐구하다"]}

        results = []
        
        # 중요도에 따라 사유를 분열시킬 렌즈의 개수도 스스로 조율 (최대 3개 이상)
        active_perspectives = list(perspectives.items())
        if importance_score < 4:
            active_perspectives = active_perspectives[:1] # 관심 없으면 1갈래만 생각함
            
        for p_name, keywords in active_perspectives:
            trajectory = [word_x]
            current_word = word_x
            
            # 자율 스케일링: 프로그래머가 정해준 range(3)이 아니라 자신의 결정(importance_score)만큼 파고듦
            for _ in range(importance_score):
                node = self.portal.word_graph.get(current_word, {})
                
                # 1. 1차원적 명시적 연결망 탐색
                conns_raw = node.get("connections", [])
                if isinstance(conns_raw, dict):
                    conns = conns_raw.get("binds_to", [])
                else:
                    conns = conns_raw
                if isinstance(conns, str): conns = [conns]
                
                next_word = None
                for conn in conns:
                    if conn in word_y or conn in word_z or word_y in conn or word_z in conn:
                        next_word = conn
                        break
                
                # 2. 명시적 연결망에 해답이 없다면, 현재 렌즈(Perspective)의 키워드를 기반으로 공명 탐색
                if not next_word:
                    cur_role = node.get("structural_role", "")
                    cur_why = node.get("why_it_exists", "")
                    
                    if cur_role or cur_why:
                        best_match = None
                        highest_resonance = 0
                        
                        for w in self.words:
                            if w == current_word or w in trajectory: continue
                            w_node = self.portal.word_graph.get(w, {})
                            w_role = w_node.get("structural_role", "")
                            w_why = w_node.get("why_it_exists", "")
                            
                            overlap = 0
                            for kw in keywords:
                                if kw in cur_role or kw in cur_why:
                                    if kw in w_role or kw in w_why:
                                        overlap += 2 # 해당 관점의 키워드가 일치하면 가중치 2배
                                        
                            # 자율적 분별력: 공명 임계치를 넘지 못하면 사유를 거부함
                            if overlap >= resonance_threshold and overlap > highest_resonance:
                                highest_resonance = overlap
                                best_match = w
                                
                        if best_match:
                            next_word = best_match
                
                if next_word and next_word not in trajectory:
                    trajectory.append(next_word)
                    current_word = next_word
                else:
                    break
                    
            # 3. 만약 사유가 막혀버렸다면 (연결성 0), 억지로 멈추지 않고 '탐구/호기심'으로 분열시킴
            if len(trajectory) == 1:
                current_word = "탐구하다"
                trajectory.append("궁금하다")
                trajectory.append("탐구하다")
                
            results.append({
                "equilibrium_word": current_word,
                "trajectory": trajectory,
                "perspective": p_name
            })
            
        return results
        
    def verify_syntactic_graph(self, graph: dict) -> dict:
        """
        파싱된 구문-의미 연결망(Graph)을 받아, 엘리시아의 사전(Lexicon) 지식과 대조합니다.
        문장 내의 '주체 -> 수식 -> 대상' 이라는 명제가 사전적 이치에 부합하는지 검증합니다.
        """
        if not self.portal:
            return {"verified": False, "reason": "사전이 존재하지 않음"}
            
        subject = graph.get("subject")
        target = graph.get("target")
        modifiers = graph.get("modifiers", [])
        
        # 주체와 대상이 모두 있어야 완결된 정의(Definition)로 검증 가능
        if not subject or not target:
            return {"verified": False, "reason": "불완전한 문장: 주체 또는 대상 누락"}
            
        s_node = self.portal.word_graph.get(subject, {})
        t_node = self.portal.word_graph.get(target, {})
        
        # 1. 대상(Target)이 Lexicon에 존재하는지 확인 (본질 파악)
        if target in self.words:
            # 주체가 미지의 단어라면, 기지의 대상을 통해 새로운 개념으로 유추 편입
            if subject not in self.words:
                self._learn_new_concept(subject, modifiers, target)
                return {
                    "verified": True, 
                    "reason": f"미지의 주체 '{subject}'를 기지의 대상 '{target}'(으)로 사전에 새롭게 정의함",
                    "learned_concept": subject,
                    "binds_to": target
                }
            else:
                # 둘 다 아는 단어라면 연결성(공명) 검증
                s_conns = s_node.get("connections", {}).get("binds_to", [])
                if isinstance(s_conns, str): s_conns = [s_conns]
                
                # 명시적 연결 확인
                if target in s_conns:
                    return {"verified": True, "reason": "사전의 명시적 연결망(binds_to)과 완벽히 공명함"}
                    
                # 수식어(Modifier)를 통한 간접 의미망 연결 확인
                for mod in modifiers:
                    if mod in str(s_node.get("structural_role","")) or mod in str(t_node.get("structural_role","")):
                        return {"verified": True, "reason": f"수식어 '{mod}'를 통해 두 개념의 속성이 교차됨"}
                        
                return {"verified": True, "reason": "새로운 관계성 편입: 알려진 두 개념의 새로운 문법적 연결"}
        else:
            return {"verified": False, "reason": f"대상 '{target}'의 본질을 사유할 수 없음 (사전에 부재)"}
            
    def _learn_new_concept(self, subject: str, modifiers: list, target: str):
        """새로운 단어를 스스로 정의하여 사전에 추가합니다."""
        if not self.portal: return
        
        # 속성들(Modifiers)과 대상(Target)을 엮어 존재 이유와 역할을 정의함
        mod_str = " ".join(modifiers)
        role_desc = f"[{target}]의 한 형태로, {mod_str} 속성을 지닌 대상."
        why_desc = f"관측된 속성({', '.join(modifiers)})을 기반으로 [{target}]의 의미망 내에 자가 편입된 새로운 인지 대상."
        
        # 대상(target)과 수식어들을 모두 연결망에 엮어둠
        binds = [target] + [m for m in modifiers if len(m) > 1]
        
        success = self.portal.add_concept(
            word=subject,
            structural_role=role_desc,
            why_it_exists=why_desc,
            binds_to=binds,
            syntactic_trajectory=f"관측 -> 속성 교차 -> {target}(으)로 정의됨"
        )
        
        if success:
            self.words = list(self.portal.word_graph.keys())
