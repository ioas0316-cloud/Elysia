"""
Korean Dictionary Dump Parser (한국어 학습 시스템)
======================================================

"API    40 +      "

    (                )         
    : https://opendict.korean.go.kr/

        : XML
-    ,    ,   ,      

[NEW 2025-12-16] API            
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
                 

    40 +        ,   ,      
                   
    """

    def __init__(self, dump_path: str = None):
        self.dump_path = Path(dump_path) if dump_path else None

        #         
        self.pos_map = {
            "  ": "Noun",
            "  ": "Verb",
            "   ": "Adjective",
            "  ": "Adverb",
            "  ": "Particle",
            "   ": "Interjection",
            "   ": "Pronoun",
            "  ": "Numeral",
            "   ": "Determiner",
            "  ": "Ending",
            "  ": "Affix",
        }

        #   
        self.total_entries = 0
        self.skipped = 0

        logger.info("  Woorimalsamam Dictionary Parser initialized")

    def set_dump_path(self, path: str):
        """        """
        self.dump_path = Path(path)
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dictionary dump not found: {path}")

    def stream_entries(self, max_entries: int = None) -> Generator[Dict[str, Any], None, None]:
        """
                  

        Yields: {
            "word": str,
            "meaning": str,
            "pos": str,
            "examples": List[str],
            "origin": str (  )
        }
        """
        if not self.dump_path or not self.dump_path.exists():
            logger.error("  No dump file set. Use set_dump_path() first.")
            return

        logger.info(f"  Streaming dictionary entries from {self.dump_path}")

        try:
            #        
            context = ET.iterparse(str(self.dump_path), events=('end',))

            for event, elem in context:
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                #           
                if tag_name in ('LexicalEntry', 'item', 'entry', 'word_info', 'Entry'):
                    entry = self._parse_entry(elem)

                    if entry and entry.get("word") and entry.get("meaning"):
                        self.total_entries += 1

                        if self.total_entries % 10000 == 0:
                            logger.info(f"     Parsed {self.total_entries} entries...")

                        yield entry

                        if max_entries and self.total_entries >= max_entries:
                            break
                    else:
                        self.skipped += 1

                    #       
                    elem.clear()

        except ET.ParseError as e:
            logger.error(f"XML Parse error: {e}")
        except Exception as e:
            logger.error(f"Error streaming entries: {e}")

        logger.info(f"  Parsing complete: {self.total_entries} entries, {self.skipped} skipped")

    def _parse_entry(self, elem) -> Optional[Dict[str, Any]]:
        """XML              """
        entry = {
            "word": "",
            "meaning": "",
            "pos": "",
            "examples": [],
            "origin": ""
        }

        #     XML      
        #    1:        
        word_elem = (
            elem.find('.//word_unit') or
            elem.find('.//word') or
            elem.find('.//headword') or
            elem.find('.//   ')
        )
        if word_elem is not None and word_elem.text:
            entry["word"] = word_elem.text.strip()

        #    
        meaning_elem = (
            elem.find('.//definition') or
            elem.find('.//meaning') or
            elem.find('.//sense_info/definition') or
            elem.find('.//   ')
        )
        if meaning_elem is not None and meaning_elem.text:
            entry["meaning"] = meaning_elem.text.strip()

        #   
        pos_elem = (
            elem.find('.//pos') or
            elem.find('.//part_of_speech') or
            elem.find('.//  ')
        )
        if pos_elem is not None and pos_elem.text:
            pos_kr = pos_elem.text.strip()
            entry["pos"] = self.pos_map.get(pos_kr, pos_kr)

        #   
        example_elems = elem.findall('.//example')
        if not example_elems:
            example_elems = elem.findall('.//  ')
        for ex_elem in example_elems:
            if ex_elem is not None and ex_elem.text:
                entry["examples"].append(ex_elem.text.strip())

        #   
        origin_elem = elem.find('.//origin')
        if origin_elem is None:
            origin_elem = elem.find('.//  ')
        if origin_elem is not None and origin_elem.text:
            entry["origin"] = origin_elem.text.strip()

        return entry if entry["word"] else None

    def absorb_to_universe(self, max_entries: int = 10000, batch_size: int = 100) -> Dict[str, int]:
        """
                InternalUniverse    
        """
        try:
            from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse
            universe = InternalUniverse()
        except Exception as e:
            logger.error(f"Failed to connect to InternalUniverse: {e}")
            return {"error": str(e)}

        results = {"absorbed": 0, "isolated": 0, "failed": 0}
        batch = []

        for entry in self.stream_entries(max_entries=max_entries):
            #           
            content = f"{entry['word']}: {entry['meaning']}"
            if entry['examples']:
                content += f"  : {entry['examples'][0]}"

            batch.append({
                "topic": f"dict:{entry['word']}",
                "content": content
            })

            if len(batch) >= batch_size:
                batch_result = universe.absorb_batch(batch)
                results["absorbed"] += batch_result.get("absorbed", 0)
                results["isolated"] += batch_result.get("isolated", 0)
                batch = []

        #      
        if batch:
            batch_result = universe.absorb_batch(batch)
            results["absorbed"] += batch_result.get("absorbed", 0)
            results["isolated"] += batch_result.get("isolated", 0)

        logger.info(f"  Dictionary absorption complete!")
        logger.info(f"   Absorbed: {results['absorbed']}, Isolated: {results['isolated']}")

        return results

    def export_to_json(self, output_path: str, max_entries: int = None) -> int:
        """JSON        (코드 베이스 구조 로터)"""
        entries = []

        for entry in self.stream_entries(max_entries=max_entries):
            entries.append(entry)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        logger.info(f"  Exported {len(entries)} entries to {output_path}")
        return len(entries)

    def create_sample_dump(self, output_path: str, count: int = 1000):
        """
                 (JSON    -          )
        """
        sample_words = [
            ("  ", "  ", "                 ", ["            "]),
            ("  ", "  ", "            ", ["           "]),
            ("  ", "  ", "               ", ["      "]),
            ("  ", "  ", "                  ", ["        "]),
            ("  ", "  ", "             ", ["     "]),
            ("  ", "  ", "                 ", ["      "]),
            ("   ", "   ", "           ", ["    "]),
            ("  ", "   ", "  ,           ", ["   "]),
            ("  ", "  ", "        ", ["      "]),
            ("  ", "  ", "             ", ["     "]),
            ("  ", "  ", "                ", ["          "]),
            ("  ", "  ", "               ", ["       "]),
            ("  ", "  ", "                  ", ["        "]),
            ("  ", "  ", "                ", ["     "]),
            ("  ", "  ", "           ", ["     "]),
        ]

        #   
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

        # JSON     
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json_path = output_path.replace('.xml', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extended_words, f, ensure_ascii=False, indent=2)

        logger.info(f"  Created sample dictionary: {json_path} ({count} entries)")
        return json_path

    def absorb_json_dictionary(self, json_path: str, max_entries: int = None) -> Dict[str, int]:
        """JSON              """
        try:
            from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse
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
                content += f"  : {entry['examples'][0]}"

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

        logger.info(f"  Dictionary absorption complete!")
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

        #      
        print("\n" + "="*60)
        print("  Absorbing sample dictionary...")
        print("="*60)

        results = dict_parser.absorb_json_dictionary(json_path, max_entries=args.sample)
        print(f"\n  Done! Absorbed {results.get('absorbed', 0)} words")

    elif args.dump:
        if args.dump.endswith('.json'):
            # JSON      
            if args.absorb:
                results = dict_parser.absorb_json_dictionary(args.dump, max_entries=args.max)
                print(f"  Absorbed {results.get('absorbed', 0)} words")
        else:
            # XML   
            dict_parser.set_dump_path(args.dump)

            if args.absorb:
                results = dict_parser.absorb_to_universe(max_entries=args.max)
                print(f"  Absorbed {results.get('absorbed', 0)} words")

            elif args.export:
                count = dict_parser.export_to_json(args.export, max_entries=args.max)
                print(f"  Exported {count} words to {args.export}")

            else:
                #     
                print("\n  Preview (first 10 entries):")
                for i, entry in enumerate(dict_parser.stream_entries(max_entries=10)):
                    print(f"   {entry['word']} ({entry['pos']}): {entry['meaning'][:50]}...")
    else:
        print("  Usage:")
        print("   --sample 500    : Generate and absorb sample dictionary")
        print("   --dump FILE     : Use dictionary dump (XML or JSON)")
        print("   --absorb        : Absorb to InternalUniverse")
        print("\n  Download real dump from: https://opendict.korean.go.kr/")
