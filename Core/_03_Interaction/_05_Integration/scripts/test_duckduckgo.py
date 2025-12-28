"""DuckDuckGo Search 테스트 - 한국어"""
import warnings
warnings.filterwarnings('ignore')

from duckduckgo_search import DDGS

query = "자유란 무엇인가"
print(f"=== DuckDuckGo 검색: {query} ===")

ddgs = DDGS()
results = list(ddgs.text(query, region='kr-kr', max_results=5))

print(f"결과 수: {len(results)}")

for i, r in enumerate(results):
    title = r.get('title', 'No title')
    body = r.get('body', '')[:80]
    print(f"\n{i+1}. {title}")
    print(f"   {body}...")
