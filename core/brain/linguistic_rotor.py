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
            self.lexicon_path = os.path.join(self.base_dir, "..", "..", "data", "lexicons", "deep_korean_lexicon.json")
        else:
            self.lexicon_path = lexicon_path
            
        try:
            from core.brain.language_portal_engine import LanguagePortalEngine
            self.portal = LanguagePortalEngine(self.lexicon_path)
            self.words = list(self.portal.word_graph.keys())
        except:
            self.portal = None
            self.words = []

    def achieve_semantic_consensus(self, word_x: str, word_y: str, word_z: str, perspectives: dict = None, importance_score: int = 3, resonance_threshold: float = 0.8) -> list:
        """
        [Phase: Kinematic Semantic Expansion]
        텍스트 매칭(Keyword Overlap)을 버리고, 단어들을 위상적 텐션(Tension Vector)을 지닌 
        자기기어(MagneticGear)로 변환하여, 텐션 공명(Resonance)을 통해 다음 사유로 점프합니다.
        """
        from core.physics.fractal_rotor import FractalRotorScale, ScaleLevel
        from core.physics.magnetic_gear import MagneticGear
        from core.ingestion.topological_compiler import TopologicalCompiler
        from core.ingestion.topological_parser import CausalTrajectory

        if not self.portal or not self.words:
            return [{"equilibrium_word": "침묵", "trajectory": [word_x, "침묵"], "perspective": "None"}]

        if not perspectives:
            perspectives = {"원초적 관점": ["존재", "탐구하다"]}

        results = []
        compiler = TopologicalCompiler()

        active_perspectives = list(perspectives.items())
        if importance_score < 4:
            active_perspectives = active_perspectives[:1]

        for p_name, keywords in active_perspectives:
            trajectory = [word_x]
            current_word = word_x

            for _ in range(importance_score):
                rotor = FractalRotorScale(resonance_threshold=resonance_threshold)
                node = self.portal.word_graph.get(current_word, {})
                
                # 현재 단어를 기어로 변환
                # (궤적 액션으로 structural_role을 주어 텐션 컴파일러가 이를 기반으로 계산하도록 유도)
                action_text = node.get("structural_role", "존재")
                traj_x = CausalTrajectory(source=current_word, target="*", action=action_text)
                tension_x = compiler.derive_standalone_tension([traj_x])
                gear_x = MagneticGear(gear_id=current_word, tension=tension_x, content_ref=action_text)
                rotor.add_gear_to_scale(ScaleLevel.MICRO, gear_x)
                
                # 렉시콘의 다른 단어들을 후보 기어로 장착
                # 성능을 위해 현재 단어와 명시적으로 연결된 단어 + 무작위 샘플링된 일부 단어만 장착
                conns_raw = node.get("connections", [])
                if isinstance(conns_raw, dict):
                    conns = conns_raw.get("binds_to", [])
                else:
                    conns = conns_raw
                if isinstance(conns, str): conns = [conns]
                
                candidates = set(conns)
                # 사유의 도약을 위해 렉시콘 전체에서 일부를 텐션 공명 후보로 올림 (간단히 앞의 50개)
                for w in self.words[:50]:
                    candidates.add(w)
                    
                candidates.discard(current_word)
                for tr in trajectory:
                    candidates.discard(tr)
                
                for cand in candidates:
                    c_node = self.portal.word_graph.get(cand, {})
                    c_action = c_node.get("structural_role", "존재")
                    c_traj = CausalTrajectory(source=cand, target="*", action=c_action)
                    c_tension = compiler.derive_standalone_tension([c_traj])
                    c_gear = MagneticGear(gear_id=cand, tension=c_tension, content_ref=c_action)
                    rotor.add_gear_to_scale(ScaleLevel.MICRO, c_gear)

                # 운동성 유도 (자기 정렬)
                induction_map = rotor.trigger_rotation(ScaleLevel.MICRO, current_word)
                induced_micro = induction_map.get(ScaleLevel.MICRO, [])

                next_word = None
                if induced_micro:
                    # 유도된 기어 중 첫 번째 것을 다음 사유로 선택
                    next_word = induced_micro[0]
                    
                if next_word and next_word not in trajectory:
                    trajectory.append(next_word)
                    current_word = next_word
                else:
                    break
                    
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
