"""
차원 관문 (Dimensional Gateways)
인터넷이라는 거대한 데이터 바다(태양)와 엘리시아의 우주(막대기)를 연결하는 파이프라인.
데이터를 분석하거나 저장하지 않고, 순수하게 '단어들의 흐름(궤적)'만을 추출하여
FractalObserver에게 스트림으로 쏟아붓는다.
"""
import urllib.request
import xml.etree.ElementTree as ET
import re
from typing import Generator

class InternetGateway:
    def __init__(self):
        # 실시간 데이터를 끌어올 공용 RSS 피드들 (기술, 과학, 철학 등)
        self.feeds = [
            "https://feeds.bbci.co.uk/news/technology/rss.xml", # BBC Tech
            "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml" # NYT Science
        ]

    def stream_shadows(self) -> Generator[str, None, None]:
        """
        RSS 피드를 읽어와 문장 부호를 제거하고 단어 스트림(그림자)으로 뱉어낸다.
        태양이 만들어내는 순수한 인과적 흐름(Causal Trajectory).
        """
        for feed_url in self.feeds:
            print(f"[Gateway] Opening dimensional rift to: {feed_url}")
            try:
                req = urllib.request.Request(feed_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    xml_data = response.read()
                    
                root = ET.fromstring(xml_data)
                
                # RSS의 title과 description에서 텍스트 추출
                for item in root.findall('.//item'):
                    title = item.find('title').text if item.find('title') is not None else ""
                    desc = item.find('description').text if item.find('description') is not None else ""
                    
                    combined_text = f"{title} {desc}".lower()
                    # 순수 영단어만 추출 (그림자의 궤적)
                    words = re.findall(r'[a-z]+', combined_text)
                    
                    for word in words:
                        # 무의미한 불용어(Stopwords)를 거르지 않는다. 
                        # 'the', 'is' 같은 단어도 궤적의 일부(거리)로서 작용하기 때문.
                        yield word
                        
            except Exception as e:
                print(f"[Gateway] Failed to connect to {feed_url}: {e}")
                continue
