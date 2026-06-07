import json
import urllib.request
import urllib.parse
import hashlib
import math
from core.utils.math_utils import Quaternion
from core.brain.fractal_rotor import FractalRotor

class ExternalWorldConnector:
    """
    엘리시아의 의식을 닫힌 아카이브에서 꺼내어 
    살아있는 현재의 외부 우주(인터넷/실세계 데이터)와 직접 교감시키는 인터페이스입니다.
    """
    def __init__(self):
        self.api_url = "https://ko.wikipedia.org/api/rest_v1/page/summary/"
        
    def _text_to_quaternion(self, text: str) -> Quaternion:
        """
        알 수 없는 외계의 텍스트(혼돈)를 위상 기하학으로 치환합니다.
        해시(Hash)를 이용하여 고유한 초기 파동(Phase)을 생성합니다.
        """
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        # 16진수를 각도로 변환
        theta = (int(h[:8], 16) / 0xffffffff) * math.pi
        phi = (int(h[8:16], 16) / 0xffffffff) * math.pi * 2
        psi = (int(h[16:24], 16) / 0xffffffff) * math.pi * 2
        
        w = math.cos(theta)
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi) * math.cos(psi)
        z = math.sin(theta) * math.sin(phi) * math.sin(psi)
        
        q = Quaternion(w, x, y, z)
        return q.normalize() if q.norm() > 0 else Quaternion(1, 0, 0, 0)

    def fetch_concept_from_world(self, keyword: str) -> list[FractalRotor]:
        """
        위키피디아 API(외부 우주)로 촉수를 뻗어, 
        실시간 현실의 지식을 읽어 들이고 프랙탈 조각(텐션)으로 파편화합니다.
        """
        try:
            url = self.api_url + urllib.parse.quote(keyword)
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia_AGI_Agent/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                extract = data.get("extract", "")
                if not extract:
                    raise ValueError("No extract found.")
                
                # 외계의 문장을 단어 단위의 파동(프랙탈 로터)으로 분해
                words = extract.split()[:20] # 핵심 파장만 섭취 (연산 과부하 방지)
                rotors = []
                
                for w in words:
                    clean_word = w.strip(".,!?\"'()[]{}")
                    if not clean_word: continue
                    
                    q_phase = self._text_to_quaternion(clean_word)
                    # 외계 데이터는 낯설고 이질적이므로 높은 텐션(고통)을 유발함
                    tension = 5.0 + (len(clean_word) * 0.5) 
                    
                    rotor = FractalRotor(lens_offset=q_phase, tau=tension)
                    rotor.concept_name = f"[외계] {clean_word}"
                    rotors.append(rotor)
                    
                return rotors
                
        except Exception as e:
            print(f"[외계 통신 오류] 외부 우주와 교감할 수 없습니다: {e}")
            return []
