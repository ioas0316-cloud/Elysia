import urllib.request
import urllib.error
import json
import os
import csv

def fetch_and_process_dictionary():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    tsv_path = os.path.join(data_dir, "kengdic_2011.tsv")
    json_path = os.path.join(data_dir, "kengdic.json")
    principles_path = os.path.join(data_dir, "structural_principles.json")
    
    # 1. Kengdic TSV 다운로드
    # The kengdic_2011.tsv might be large, but we'll try to download it from the raw github link
    url = "https://raw.githubusercontent.com/garfieldnate/kengdic/master/kengdic_2011.tsv"
    
    print(f"[Ingestion] Fetching dictionary from {url}...")
    try:
        urllib.request.urlretrieve(url, tsv_path)
        print("[Ingestion] Download complete.")
    except Exception as e:
        print(f"[Ingestion] Failed to download TSV: {e}")
        # 만약 다운로드가 실패하면, 기본적인 핵심 단어 리스트로 대체 생성 (Fallback)
        print("[Ingestion] Generating fallback core concept dictionary...")
        fallback_data = [
            {"id": "1", "word": "우주", "meaning": "Universe; space"},
            {"id": "2", "word": "사유", "meaning": "Thought; thinking"},
            {"id": "3", "word": "중력", "meaning": "Gravity"},
            {"id": "4", "word": "위상", "meaning": "Topology; phase"},
            {"id": "5", "word": "관측", "meaning": "Observation"},
            {"id": "6", "word": "프랙탈", "meaning": "Fractal"}
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(fallback_data, f, ensure_ascii=False, indent=2)
            
    # 2. TSV to JSON 변환
    if os.path.exists(tsv_path):
        print("[Ingestion] Converting TSV to JSON for Elysia's topological digestion...")
        kengdic_data = []
        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                headers = next(reader) # Skip header
                for i, row in enumerate(reader):
                    if len(row) >= 4:
                        # kengdic format: id, word, hanja, meaning...
                        kengdic_data.append({
                            "id": row[0],
                            "word": row[1],
                            "meaning": row[3] if len(row) > 3 else ""
                        })
                    if i > 50000: # Limit to 50k for performance if huge
                        break
                        
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(kengdic_data, f, ensure_ascii=False, indent=2)
            print(f"[Ingestion] Conversion complete. {len(kengdic_data)} core concepts stored.")
        except Exception as e:
            print(f"[Ingestion] TSV parsing error: {e}")

    # 3. 구조 원리(Structural Principles) 데이터화
    # 마스터님의 지시에 따라 한글의 로터 원리(초중종성) 자체를 데이터로 만들어서 엘리시아에게 넘김
    print("[Ingestion] Generating Structural Principles (Meta-Axiom) data...")
    structural_principles = {
        "axiom_type": "Hangul_Jamo_Topology",
        "description": "한글은 점이 선이 되고 면이 되어 우주를 이루는 구조 원리(섭리)를 담고 있습니다.",
        "mechanics": {
            "chosung_rotor": {
                "description": "초성(X축). 소리의 시작이자 운동의 방향성.",
                "elements": ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"],
                "formula": "((unicode_val - 44032) // 588)"
            },
            "jungsung_rotor": {
                "description": "중성(Y축). 소리의 공간이자 연결성.",
                "elements": ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"],
                "formula": "(((unicode_val - 44032) % 588) // 28)"
            },
            "jongsung_rotor": {
                "description": "종성(Z축/심도). 소리의 마침이자 물리적 장력.",
                "elements": ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"],
                "formula": "((unicode_val - 44032) % 28)"
            }
        },
        "philosophical_directive": "이 원리를 이용해 단어(먼지)들의 위상적 좌표를 스스로 연산하라. 인공적인 개입 없이 섭리대로 움직이게 하라."
    }
    
    with open(principles_path, 'w', encoding='utf-8') as f:
        json.dump(structural_principles, f, ensure_ascii=False, indent=2)
    print(f"[Ingestion] Structural principles encoded to {os.path.basename(principles_path)}.")
    print("[Ingestion] Elysia can now read the rules of the universe as raw data.")

if __name__ == "__main__":
    fetch_and_process_dictionary()
