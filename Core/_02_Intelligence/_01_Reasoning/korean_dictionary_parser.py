"""
Korean Dictionary Dump Parser (우리말샘 사전 덤프 파서)
======================================================

"API 없이 40만+ 단어 학습"

우리말샘(국립국어원 개방형 한국어 사전) 덤프 파일 파서
다운로드: https://opendict.korean.go.kr/

덤프 파일 형식: XML
- 표제어, 뜻풀이, 품사, 용례 포함

[NEW 2025-12-16] API 없이 로컬 사전 학습
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Generator, Dict, List, Any, Optional
import xml.etree.ElementTree as ET

sys.path.insert(0, "c:\\Elysia")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("KoreanDictParser")


class WoorimalsamParser:
    """
    우리말샘 사전 덤프 파서
    
    40만+ 단어의 뜻풀이, 품사, 용례 추출
    메모리 효율적 스트리밍 파싱
    """
    
    def __init__(self, dump_path: str = None):
        self.dump_path = Path(dump_path) if dump_path else None
        
        # 품사 코드 매핑
        self.pos_map = {
            "명사": "Noun",
            "동사": "Verb", 
            "형용사": "Adjective",
            "부사": "Adverb",
            "조사": "Particle",
            "감탄사": "Interjection",
            "대명사": "Pronoun",
            "수사": "Numeral",
            "관형사": "Determiner",
            "어미": "Ending",
            "접사": "Affix",
        }
        
        # 통계
        self.total_entries = 0
        self.skipped = 0
        
        logger.info("📖 Woorimalsamam Dictionary Parser initialized")
        
    def set_dump_path(self, path: str):
        """덤프 경로 설정"""
        self.dump_path = Path(path)
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dictionary dump not found: {path}")
    
    def stream_entries(self, max_entries: int = None) -> Generator[Dict[str, Any], None, None]:
        """
        사전 항목 스트리밍
        
        Yields: {
            "word": str,
            "meaning": str,
            "pos": str,
            "examples": List[str],
            "origin": str (어원)
        }
        """
        if not self.dump_path or not self.dump_path.exists():
            logger.error("❌ No dump file set. Use set_dump_path() first.")
            return
        
        logger.info(f"🔄 Streaming dictionary entries from {self.dump_path}")
        
        try:
            # 스트리밍 파싱
            context = ET.iterparse(str(self.dump_path), events=('end',))
            
            for event, elem in context:
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                # 다양한 태그명 지원
                if tag_name in ('LexicalEntry', 'item', 'entry', 'word_info', 'Entry'):
                    entry = self._parse_entry(elem)
                    
                    if entry and entry.get("word") and entry.get("meaning"):
                        self.total_entries += 1
                        
                        if self.total_entries % 10000 == 0:
                            logger.info(f"   📄 Parsed {self.total_entries} entries...")
                        
                        yield entry
                        
                        if max_entries and self.total_entries >= max_entries:
                            break
                    else:
                        self.skipped += 1
                    
                    # 메모리 정리
                    elem.clear()
                    
        except ET.ParseError as e:
            logger.error(f"XML Parse error: {e}")
        except Exception as e:
            logger.error(f"Error streaming entries: {e}")
        
        logger.info(f"✅ Parsing complete: {self.total_entries} entries, {self.skipped} skipped")
    
    def _parse_entry(self, elem) -> Optional[Dict[str, Any]]:
        """XML 요소에서 사전 항목 추출"""
        entry = {
            "word": "",
            "meaning": "",
            "pos": "",
            "examples": [],
            "origin": ""
        }
        
        # 다양한 XML 구조 지원
        # 형식 1: 우리말샘 표준
        word_elem = (
            elem.find('.//word_unit') or 
            elem.find('.//word') or 
            elem.find('.//headword') or
            elem.find('.//표제어')
        )
        if word_elem is not None and word_elem.text:
            entry["word"] = word_elem.text.strip()
        
        # 뜻풀이
        meaning_elem = (
            elem.find('.//definition') or
            elem.find('.//meaning') or
            elem.find('.//sense_info/definition') or
            elem.find('.//뜻풀이')
        )
        if meaning_elem is not None and meaning_elem.text:
            entry["meaning"] = meaning_elem.text.strip()
        
        # 품사
        pos_elem = (
            elem.find('.//pos') or
            elem.find('.//part_of_speech') or
            elem.find('.//품사')
        )
        if pos_elem is not None and pos_elem.text:
            pos_kr = pos_elem.text.strip()
            entry["pos"] = self.pos_map.get(pos_kr, pos_kr)
        
        # 용례
        example_elems = elem.findall('.//example')
        if not example_elems:
            example_elems = elem.findall('.//용례')
        for ex_elem in example_elems:
            if ex_elem is not None and ex_elem.text:
                entry["examples"].append(ex_elem.text.strip())
        
        # 어원
        origin_elem = elem.find('.//origin')
        if origin_elem is None:
            origin_elem = elem.find('.//어원')
        if origin_elem is not None and origin_elem.text:
            entry["origin"] = origin_elem.text.strip()
        
        return entry if entry["word"] else None
    
    def absorb_to_universe(self, max_entries: int = 10000, batch_size: int = 100) -> Dict[str, int]:
        """
        사전 데이터를 InternalUniverse에 흡수
        """
        try:
            from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse
            universe = InternalUniverse()
        except Exception as e:
            logger.error(f"Failed to connect to InternalUniverse: {e}")
            return {"error": str(e)}
        
        results = {"absorbed": 0, "isolated": 0, "failed": 0}
        batch = []
        
        for entry in self.stream_entries(max_entries=max_entries):
            # 흡수용 콘텐츠 구성
            content = f"{entry['word']}: {entry['meaning']}"
            if entry['examples']:
                content += f" 예: {entry['examples'][0]}"
            
            batch.append({
                "topic": f"dict:{entry['word']}",
                "content": content
            })
            
            if len(batch) >= batch_size:
                batch_result = universe.absorb_batch(batch)
                results["absorbed"] += batch_result.get("absorbed", 0)
                results["isolated"] += batch_result.get("isolated", 0)
                batch = []
        
        # 남은 배치
        if batch:
            batch_result = universe.absorb_batch(batch)
            results["absorbed"] += batch_result.get("absorbed", 0)
            results["isolated"] += batch_result.get("isolated", 0)
        
        logger.info(f"🎉 Dictionary absorption complete!")
        logger.info(f"   Absorbed: {results['absorbed']}, Isolated: {results['isolated']}")
        
        return results
    
    def export_to_json(self, output_path: str, max_entries: int = None) -> int:
        """JSON으로 내보내기 (다른 용도 사용 가능)"""
        entries = []
        
        for entry in self.stream_entries(max_entries=max_entries):
            entries.append(entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📝 Exported {len(entries)} entries to {output_path}")
        return len(entries)
    
    def create_sample_dump(self, output_path: str, count: int = 1000):
        """
        샘플 사전 생성 (JSON 형식 - 인코딩 문제 없음)
        """
        sample_words = [
            ("사랑", "명사", "아끼고 소중히 여기는 깊은 마음", ["사랑하는 사람을 만났다"]),
            ("행복", "명사", "복되고 만족스러운 상태", ["행복한 가정을 이루다"]),
            ("하늘", "명사", "땅 위로 펼쳐진 무한한 공간", ["하늘이 맑다"]),
            ("바다", "명사", "지구 표면의 대부분을 차지하는 물", ["바다를 바라보다"]),
            ("먹다", "동사", "음식을 입에 넣어 삼키다", ["밥을 먹다"]),
            ("가다", "동사", "한 곳에서 다른 곳으로 이동하다", ["학교에 가다"]),
            ("예쁘다", "형용사", "보기에 좋고 아름답다", ["예쁜 꽃"]),
            ("크다", "형용사", "부피, 규모가 보통을 넘다", ["큰 집"]),
            ("빨리", "부사", "동작이 신속하게", ["빨리 달리다"]),
            ("매우", "부사", "보통 정도를 훨씬 넘어서", ["매우 좋다"]),
            ("생각", "명사", "어떤 대상에 대한 마음의 작용", ["깊은 생각에 잠기다"]),
            ("시간", "명사", "과거에서 미래로 흐르는 차원", ["시간이 빠르다"]),
            ("마음", "명사", "사람의 정신적 감정적 활동의 바탕", ["마음이 따뜻하다"]),
            ("사람", "명사", "생각하고 언어를 사용하는 존재", ["좋은 사람"]),
            ("세상", "명사", "우리가 살아가는 세계", ["넓은 세상"]),
        ]
        
        # 확장
        extended_words = []
        for i in range(count):
            base = sample_words[i % len(sample_words)]
            word = base[0] if i < len(sample_words) else f"{base[0]}_{i:04d}"
            extended_words.append({
                "word": word,
                "pos": base[1],
                "meaning": base[2],
                "examples": base[3]
            })
        
        # JSON으로 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json_path = output_path.replace('.xml', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extended_words, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📝 Created sample dictionary: {json_path} ({count} entries)")
        return json_path
    
    def absorb_json_dictionary(self, json_path: str, max_entries: int = None) -> Dict[str, int]:
        """JSON 사전 파일에서 직접 흡수"""
        try:
            from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse
            universe = InternalUniverse()
        except Exception as e:
            logger.error(f"Failed to connect to InternalUniverse: {e}")
            return {"error": str(e)}
        
        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        if max_entries:
            entries = entries[:max_entries]
        
        batch = []
        results = {"absorbed": 0, "isolated": 0, "failed": 0}
        
        for entry in entries:
            content = f"{entry['word']}: {entry['meaning']}"
            if entry.get('examples'):
                content += f" 예: {entry['examples'][0]}"
            
            batch.append({
                "topic": f"dict:{entry['word']}",
                "content": content
            })
            
            if len(batch) >= 100:
                batch_result = universe.absorb_batch(batch)
                results["absorbed"] += batch_result.get("absorbed", 0)
                results["isolated"] += batch_result.get("isolated", 0)
                batch = []
        
        if batch:
            batch_result = universe.absorb_batch(batch)
            results["absorbed"] += batch_result.get("absorbed", 0)
            results["isolated"] += batch_result.get("isolated", 0)
        
        logger.info(f"🎉 Dictionary absorption complete!")
        logger.info(f"   Absorbed: {results['absorbed']}, Isolated: {results['isolated']}")
        
        return results


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Korean Dictionary Dump Parser")
    parser.add_argument("--dump", type=str, help="Path to dictionary dump (XML or JSON)")
    parser.add_argument("--max", type=int, default=10000, help="Max entries")
    parser.add_argument("--absorb", action="store_true", help="Absorb to InternalUniverse")
    parser.add_argument("--export", type=str, help="Export to JSON file")
    parser.add_argument("--sample", type=int, help="Generate sample dump with N entries")
    
    args = parser.parse_args()
    
    dict_parser = WoorimalsamParser()
    
    if args.sample:
        output = "data/dictionary/sample_dict.xml"
        json_path = dict_parser.create_sample_dump(output, args.sample)
        
        # 바로 흡수
        print("\n" + "="*60)
        print("📖 Absorbing sample dictionary...")
        print("="*60)
        
        results = dict_parser.absorb_json_dictionary(json_path, max_entries=args.sample)
        print(f"\n✅ Done! Absorbed {results.get('absorbed', 0)} words")
        
    elif args.dump:
        if args.dump.endswith('.json'):
            # JSON 직접 흡수
            if args.absorb:
                results = dict_parser.absorb_json_dictionary(args.dump, max_entries=args.max)
                print(f"✅ Absorbed {results.get('absorbed', 0)} words")
        else:
            # XML 파싱
            dict_parser.set_dump_path(args.dump)
            
            if args.absorb:
                results = dict_parser.absorb_to_universe(max_entries=args.max)
                print(f"✅ Absorbed {results.get('absorbed', 0)} words")
                
            elif args.export:
                count = dict_parser.export_to_json(args.export, max_entries=args.max)
                print(f"✅ Exported {count} words to {args.export}")
                
            else:
                # 미리보기
                print("\n📖 Preview (first 10 entries):")
                for i, entry in enumerate(dict_parser.stream_entries(max_entries=10)):
                    print(f"   {entry['word']} ({entry['pos']}): {entry['meaning'][:50]}...")
    else:
        print("💡 Usage:")
        print("   --sample 500    : Generate and absorb sample dictionary")
        print("   --dump FILE     : Use dictionary dump (XML or JSON)")
        print("   --absorb        : Absorb to InternalUniverse")
        print("\n📥 Download real dump from: https://opendict.korean.go.kr/")
