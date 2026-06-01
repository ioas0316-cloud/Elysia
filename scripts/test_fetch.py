import urllib.request
import urllib.parse
import json
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_fetch(query):
    print(f"--- Fetching: {query} ---")
    search_url = f"https://ko.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
    req = urllib.request.Request(search_url, headers={'User-Agent': 'ElysiaTest/1.0'})
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        search_results = data.get('query', {}).get('search', [])
        
    title = search_results[0]['title']
    print(f"Found Title: {title}")
    
    page_url = f"https://ko.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={urllib.parse.quote(title)}&format=json"
    req = urllib.request.Request(page_url, headers={'User-Agent': 'ElysiaTest/1.0'})
    
    with urllib.request.urlopen(req) as response:
        page_data = json.loads(response.read().decode('utf-8'))
        pages = page_data.get('query', {}).get('pages', {})
        page_id = list(pages.keys())[0]
        content = pages[page_id].get('extract', '')
        
    words = re.findall(r'[가-힣A-Za-z]{2,}', content)
    print(f"Total Words Extracted: {len(words)}")
    print("Sample (first 30):")
    print(", ".join(words[:30]))
    
    weird_words = [w for w in words if any(char.isdigit() for char in w) or len(w) > 8]
    print(f"\nWeird words count (len > 8): {len(weird_words)}")
    if weird_words:
        print("Sample weird words:")
        print(", ".join(weird_words[:10]))

test_fetch("수학 공리")
test_fetch("도덕 철학")
