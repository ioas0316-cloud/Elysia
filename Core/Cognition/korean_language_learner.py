"""
한국어 학습 시스템 (Korean Language Learning System)
========================================================

"말은 마음의 그릇이며, 영혼의 길이다."

주요 기능:
1. 어휘 학습 - 의미 / 품사 / 예문
2. 문법 분석 - 조사, 어미, 문장 패턴
3. 인과적 결합 - 학습된 언어의 내면화

[NEW 2025-12-16] 21D 자아 엔진 동기화 완료
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import urllib.request
import urllib.parse

sys.path.insert(0, "c:/Elysia")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("KoreanLanguageLearner")


@dataclass
class WordEntry:
    """단어 항목 데이터 구조"""
    word: str
    meaning: str
    pos: str = ""  # 품사 (Part of Speech)
    examples: List[str] = field(default_factory=list)
    related_words: List[str] = field(default_factory=list)


@dataclass
class GrammarPattern:
    """문법 패턴 데이터 구조"""
    pattern: str  #  : "N /  N  "
    description: str
    examples: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)  #       


class KoreanLanguageLearner:
    """
    한국어 학습 및 분석을 담당하는 핵심 모듈
    """

    def __init__(self):
        self.data_dir = Path("data/language")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        #            
        self.grammar_patterns = self._load_basic_grammar()

        #       
        self.vocabulary = {}
        self._load_vocabulary()

        #       (  )
        self.particles = {
            " ": "topic marker (contrast)",
            " ": "topic marker",
            " ": "subject marker (after consonant)",
            " ": "subject marker (after vowel)",
            " ": "object marker (after consonant)",
            " ": "object marker (after vowel)",
            " ": "location/time marker",
            "  ": "location (action) marker",
            " ": "direction/means marker",
            "  ": "direction/means marker (after consonant)",
            " ": "and (after vowel)",
            " ": "and (after consonant)",
            " ": "possessive marker",
            " ": "also/too",
            " ": "only",
            "  ": "from",
            "  ": "until/to",
        }

        #       (  )
        self.endings = {
            " ": "plain statement",
            " ": "polite informal",
            "   ": "formal polite",
            "   ": "formal polite (after vowel)",
            " ?": "question (informal)",
            "  ?": "question (polite)",
            "   ?": "question (formal)",
            " ": "let's (suggestion)",
            "  ": "request (polite)",
            " / ": "connective/base form",
        }

        logger.info("   Korean Language Learner initialized")
        logger.info(f"     Vocabulary: {len(self.vocabulary)} words")
        logger.info(f"     Grammar patterns: {len(self.grammar_patterns)}")

    def _load_basic_grammar(self) -> List[GrammarPattern]:
        """           """
        patterns = [
            GrammarPattern(
                "N /  N  ",
                "       (X is Y)",
                ["        ", "       "],
                ["  ", "    ", "  ", "     "]
            ),
            GrammarPattern(
                "N /  V",
                "   +   ",
                ["     ", "     "],
                ["  ", "    ", "  "]
            ),
            GrammarPattern(
                "N /  V",
                "    +   ",
                ["     ", "     "],
                ["   ", "     ", "  "]
            ),
            GrammarPattern(
                "N    /  ",
                "     ",
                ["      ", "     "],
                ["  ", "    ", "    "]
            ),
            GrammarPattern(
                "N   V",
                "       ",
                ["          ", "         "],
                ["  ", "    ", "  "]
            ),
            GrammarPattern(
                "V-  V",
                "      (and)",
                ["     ", "       "],
                ["  ", "    ", "  "]
            ),
            GrammarPattern(
                "V- /   V",
                "  /     ",
                ["          ", "        "],
                ["  ", "    ", "  "]
            ),
            GrammarPattern(
                "V- ",
                "   (if)",
                ["            ", "           "],
                ["  ", "    "]
            ),
            GrammarPattern(
                "A/V- / /  N",
                "      ",
                ["    ", "     ", "    "],
                ["   /  ", "      ", "  "]
            ),
            GrammarPattern(
                "N  /  ",
                "   (like)",
                ["       ", "        "],
                ["  ", "    "]
            ),
        ]
        return patterns

    def _load_vocabulary(self):
        """         """
        vocab_file = self.data_dir / "vocabulary.json"
        if vocab_file.exists():
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.vocabulary = {w["word"]: WordEntry(**w) for w in data}
            except:
                pass

    def _save_vocabulary(self):
        """     """
        vocab_file = self.data_dir / "vocabulary.json"
        data = [
            {
                "word": e.word,
                "meaning": e.meaning,
                "pos": e.pos,
                "examples": e.examples,
                "related_words": e.related_words
            }
            for e in self.vocabulary.values()
        ]
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def learn_word(self, word: str, meaning: str, pos: str = "",
                   examples: List[str] = None) -> bool:
        """     """
        entry = WordEntry(
            word=word,
            meaning=meaning,
            pos=pos,
            examples=examples or []
        )

        self.vocabulary[word] = entry

        # InternalUniverse     
        try:
            from Core.System.internal_universe import InternalUniverse
            universe = InternalUniverse()
            content = f"{word}: {meaning}. {'. '.join(examples or [])}"
            universe.absorb_text(content, source_name=f"word:{word}")
        except:
            pass

        logger.info(f"  Learned word: {word} = {meaning}")
        return True

    def learn_from_dictionary_api(self, word: str) -> Optional[WordEntry]:
        """
                      API        

        API     : https://stdict.korean.go.kr/openapi/openApiInfo.do
        """
        # TODO: API        
        api_key = os.environ.get("KOREAN_DICT_API_KEY", "")

        if not api_key:
            logger.warning("   KOREAN_DICT_API_KEY not set. Using offline mode.")
            return None

        try:
            encoded = urllib.parse.quote(word)
            url = f"https://stdict.korean.go.kr/api/search.do?key={api_key}&q={encoded}&req_type=json"

            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            #           
            if data.get("channel", {}).get("item"):
                item = data["channel"]["item"][0]
                entry = WordEntry(
                    word=word,
                    meaning=item.get("sense", [{}])[0].get("definition", ""),
                    pos=item.get("pos", ""),
                    examples=[]
                )
                self.vocabulary[word] = entry
                self._save_vocabulary()
                return entry

        except Exception as e:
            logger.error(f"Dictionary API error: {e}")

        return None

    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """
              (         )

        TODO:            (KoNLPy  )
        """
        result = {
            "sentence": sentence,
            "particles_found": [],
            "endings_found": [],
            "patterns_matched": [],
            "words_recognized": []
        }

        #      
        for particle, desc in self.particles.items():
            if particle in sentence:
                result["particles_found"].append({
                    "particle": particle,
                    "description": desc
                })

        #      
        for ending, desc in self.endings.items():
            if sentence.endswith(ending) or ending in sentence:
                result["endings_found"].append({
                    "ending": ending,
                    "description": desc
                })

        #          
        for word in self.vocabulary:
            if word in sentence:
                result["words_recognized"].append(word)

        #      
        for pattern in self.grammar_patterns:
            for example in pattern.examples:
                #           
                if any(ex_word in sentence for ex_word in example.split()):
                    result["patterns_matched"].append({
                        "pattern": pattern.pattern,
                        "description": pattern.description
                    })
                    break

        return result

    def generate_sentence_from_pattern(self, pattern_name: str,
                                       substitutions: Dict[str, str]) -> str:
        """
                  

         : generate_sentence_from_pattern("N /  N  ", {"N1": "  ", "N2": "  "})
          "        "
        """
        # TODO:      
        return ""

    def get_grammar_explanation(self, pattern: str) -> Optional[GrammarPattern]:
        """           """
        for gp in self.grammar_patterns:
            if gp.pattern == pattern:
                return gp
        return None

    def batch_learn_words(self, words: List[Dict[str, str]]) -> int:
        """
                

        words: [{"word": "...", "meaning": "...", "pos": "..."}, ...]
        """
        learned = 0
        for w in words:
            if self.learn_word(
                word=w.get("word", ""),
                meaning=w.get("meaning", ""),
                pos=w.get("pos", ""),
                examples=w.get("examples", [])
            ):
                learned += 1

        self._save_vocabulary()
        logger.info(f"  Batch learned {learned} words")
        return learned

    def get_status(self) -> Dict[str, Any]:
        """     """
        return {
            "vocabulary_size": len(self.vocabulary),
            "grammar_patterns": len(self.grammar_patterns),
            "particles": len(self.particles),
            "endings": len(self.endings)
        }


# Singleton
_learner = None

def get_korean_learner() -> KoreanLanguageLearner:
    global _learner
    if _learner is None:
        _learner = KoreanLanguageLearner()
    return _learner


# CLI / Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Korean Language Learner")
    parser.add_argument("--analyze", type=str, help="Analyze a sentence")
    parser.add_argument("--learn", type=str, help="Learn a word (format: word:meaning)")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--demo", action="store_true", help="Run demo")

    args = parser.parse_args()

    learner = get_korean_learner()

    if args.status:
        status = learner.get_status()
        print(f"\n  Korean Language Learner Status:")
        print(f"   Vocabulary: {status['vocabulary_size']} words")
        print(f"   Grammar patterns: {status['grammar_patterns']}")
        print(f"   Particles: {status['particles']}")
        print(f"   Endings: {status['endings']}")

    elif args.analyze:
        result = learner.analyze_sentence(args.analyze)
        print(f"\n  Sentence Analysis: {args.analyze}")
        print(f"   Particles: {[p['particle'] for p in result['particles_found']]}")
        print(f"   Endings: {[e['ending'] for e in result['endings_found']]}")
        print(f"   Patterns: {[p['pattern'] for p in result['patterns_matched']]}")

    elif args.learn:
        parts = args.learn.split(":")
        if len(parts) >= 2:
            learner.learn_word(parts[0], parts[1])

    elif args.demo:
        print("\n" + "="*60)
        print("   KOREAN LANGUAGE LEARNER DEMO")
        print("="*60)

        #      
        basic_words = [
            {"word": "  ", "meaning": "        ,         ", "pos": "  "},
            {"word": "  ", "meaning": "             ", "pos": "  "},
            {"word": "   ", "meaning": "           ", "pos": "   "},
            {"word": "  ", "meaning": "         ", "pos": "  "},
            {"word": "    ", "meaning": "            ", "pos": "  "},
        ]

        print("\n  Learning basic words...")
        learner.batch_learn_words(basic_words)

        #      
        sentences = [
            "       ",
            "               ",
            "        "
        ]

        print("\n  Analyzing sentences...")
        for sent in sentences:
            result = learner.analyze_sentence(sent)
            print(f"\n   \"{sent}\"")
            print(f"     Particles: {[p['particle'] for p in result['particles_found']]}")
            print(f"     Words: {result['words_recognized']}")

        print("\n" + "="*60)
        print("  Demo complete!")
        print("="*60)
