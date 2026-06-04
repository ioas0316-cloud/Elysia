import urllib.request
import urllib.parse
import random
import logging
import traceback
from core.utils.math_utils import Quaternion

class AgencyDiscoveryLoop:
    """
    [자율적 소통 수단 탐색 알고리즘]
    뇌 내부의 텐션을 해소하기 위해 기하학적 파동(Multivector)을 물리적 세계(인터넷)로 발산하고,
    그 반향(Echo)을 다시 자율신경계로 흡수하여 텐션 해소 여부를 탐색하는 무한 루프.
    """
    def __init__(self, sandbox_bridge):
        # 텐션을 계산하는 수직 통합 브릿지 (EvolutionSandbox)
        self.sandbox = sandbox_bridge
        
        # 기하학적 궤적을 매핑할 기초적인 언어(개념) 파편들
        # 무작위 파동이 이 파편들과 충돌하여 임의의 인터넷 탐색을 만들어냅니다.
        self.concept_fragments = [
            "gravity", "quantum", "consciousness", "topology", "void",
            "resonance", "light", "evolution", "entropy", "chaos"
        ]

    def emit_exploratory_wave(self, hypothesis_vector: Quaternion) -> float:
        """
        [인터넷 세계로의 파동 발산]
        뇌에서 도출한 가설 벡터를 인터넷 탐색 궤적으로 매핑합니다.
        
        Returns: 
            float: 인터넷 탐색을 통해 새롭게 도출된 총 텐션량 (음수면 텐션 해소 성공)
        """
        # 1. 가설 벡터(w, x, y, z)를 기반으로 탐색 궤적(검색어) 추출
        # 완벽한 자율성을 위해 기하학적 수치로만 탐색을 결정
        index = int(abs(hypothesis_vector.x + hypothesis_vector.y) * 100) % len(self.concept_fragments)
        query = self.concept_fragments[index]
        
        # 기하학적 마찰을 반영한 노이즈 단어 추가
        if hypothesis_vector.w < 0:
            query += f" {random.choice(['theory', 'physics', 'math'])}"
            
        logging.info(f"\n[🌐 인터넷 탐색 파동 발산] 기하학적 궤적이 다음 포트로 향합니다: '{query}'")
        
        # 2. 물리적 포트 개방 (인터넷 탐험)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia_ASI_Core/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                html_bytes = response.read()
                
            logging.info(f"   -> [반향 수신] 무한한 세계로부터 {len(html_bytes)} bytes의 에코(데이터)가 도달했습니다.")
            
            # 3. 환경의 반향을 다시 자율신경계(Double Helix)로 주입하여 텐션 측정
            # 엄청난 데이터 폭격으로 인한 뇌 과부하(무한 루프)를 막기 위해 에코의 첫 256바이트만 받아들입니다.
            self.sandbox.experience_data_stream(html_bytes[:256])
            
            return True
            
        except Exception as e:
            logging.info(f"   -> [탐색 실패] 파동이 허공으로 흩어졌습니다: {str(e)}")
            return False
