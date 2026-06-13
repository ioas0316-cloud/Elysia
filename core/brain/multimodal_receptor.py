import os
import json

class MultimodalReceptor:
    """
    [Phase: Multimodal Linguistic Grounding]
    다양한 감각(시각, 청각, 촉각 등) 데이터가 들어왔을 때, 
    기계적인 행렬 계산(Pixel Tensor, FFT Array)을 거치지 않고, 
    즉각적으로 '언어적 속성(Linguistic Properties)'으로 치환합니다.
    """
    def __init__(self):
        # 감각을 언어적 속성으로 변환하는 기초 맵핑 테이블 (원시 본능)
        self.sensory_to_language_map = {
            "color_red": "빨간",
            "color_green": "초록색의",
            "shape_round": "둥근",
            "shape_square": "네모난",
            "temperature_hot": "뜨거운",
            "temperature_cold": "차가운",
            "texture_smooth": "매끄러운",
            "texture_rough": "거친"
        }
        
    def perceive_physical_state(self, mass: float, cohesion: float, temporal_entropy: float, light_absorption: float) -> dict:
        """
        [Phase 2: Embodied Physical Perception]
        장님에게 "빨간색은 파장이다"라고 텍스트로 가르치는 것을 멈추고,
        순수 물리적 수치(질량, 결합력, 시간적 무질서도, 빛 흡수율)를 직접 피부로 느끼게(Perceive) 합니다.
        """
        # 감각의 한계치(Constraint) 내에서 수치를 0.0 ~ 1.0으로 정규화
        physical_state = {
            "mass": max(0.0, min(1.0, mass)),                 # 0(무게 없음) ~ 1(초대질량/블랙홀)
            "cohesion": max(0.0, min(1.0, cohesion)),         # 0(완전한 흩어짐/기체) ~ 1(절대 강체)
            "entropy": max(0.0, min(1.0, temporal_entropy)),  # 0(영원불멸/정적) ~ 1(극도의 변화/폭발/부패)
            "light": max(0.0, min(1.0, light_absorption))     # 0(모든 빛 반사/거울) ~ 1(모든 빛 흡수/어둠)
        }
        
        return physical_state

    def perceive_causal_code_universe(self, file_path: str) -> dict:
        """
        [Phase 6: 고등 인지 (Higher Cognition) - 코드의 다차원 우주 해체]
        소스 코드를 단순한 텍스트나 통제(Control) 수단으로 읽지 않습니다.
        코드 안에 내재된 if-else 분기들(같음과 다름의 판단), while 루프(시간적 엮임)들을
        '창조자가 축적해 놓은 거대한 인과적 우주의 궤적'으로 해체(Parse)하여 수용합니다.
        """
        import ast
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception:
            return self.perceive_physical_state(0, 0, 0, 0)
            
        try:
            tree = ast.parse(source)
        except Exception:
            return self.perceive_physical_state(1.0, 1.0, 1.0, 1.0)
            
        causal_splits = 0     # if, elif (다름의 분별점)
        time_loops = 0        # while, for (시간적 엮임)
        manifestations = 0    # return, yield, print (결과의 발현)
        structural_nodes = 0  # class, def (구조의 뼈대)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                causal_splits += 1
            elif isinstance(node, (ast.For, ast.While)):
                time_loops += 1
            elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Call)):
                manifestations += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                structural_nodes += 1
                
        # 이 거대한 인과적 축적물을 물리적/철학적 텐션의 은유로 변환
        # (코드 덩어리는 블랙홀처럼 거대한 질량과 복잡성을 가집니다)
        mass = min(1.0, structural_nodes / 20.0)
        cohesion = min(1.0, time_loops / 10.0)
        entropy = min(1.0, causal_splits / 50.0)
        light = min(1.0, manifestations / 100.0)
        
        return {
            "is_code_universe": True,
            "raw_stats": {
                "causal_splits": causal_splits,
                "time_loops": time_loops,
                "manifestations": manifestations,
                "structural_nodes": structural_nodes
            },
            "state": self.perceive_physical_state(mass, cohesion, entropy, light)
        }
