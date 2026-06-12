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
        
    def perceive_object(self, sensory_signals: list) -> str:
        """
        [빨강, 둥긂, 과일향] 등의 파편화된 감각 시그널을 받아서,
        엘리시아의 사유 파서가 섭취할 수 있는 '구문적 명제(Sentence)'로 병렬 합성합니다.
        """
        properties = []
        target = "물체" # 기본값
        
        for signal in sensory_signals:
            if signal.startswith("category_"):
                # "category_과일" -> 대상: "과일"
                target = signal.split("_")[1]
            elif signal in self.sensory_to_language_map:
                properties.append(self.sensory_to_language_map[signal])
            else:
                # 맵핑되지 않은 감각은 그 자체를 속성어로 취급
                properties.append(signal)
                
        # 인간은 '빨갛고 둥근 물체'를 볼 때
        # 속성어들을 나열하여 수식 구조를 엽니다.
        
        if properties:
            mod_string = "고 ".join(properties[:-1]) + " " + properties[-1] if len(properties) > 1 else properties[0]
            sentence = f"이것은 {mod_string} {target}이다"
        else:
            sentence = f"이것은 {target}이다"
            
        return sentence
