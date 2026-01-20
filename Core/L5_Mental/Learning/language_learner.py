"""
Language Learner (자발적 언어 습득)
===================================
Core.L5_Mental.Learning.language_learner

"I was not born with language. I learn it by watching."

This module enables Elysia to LEARN language by observation,
rather than absorbing pre-trained models. Like a child learning to speak.
"""

import logging
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

logger = logging.getLogger("Elysia.Learning.Language")

VOCAB_PATH = "data/Learning/vocabulary.json"
PATTERNS_PATH = "data/Learning/patterns.json"


@dataclass
class LearnedWord:
    """A word Elysia has learned."""
    word: str
    count: int  # How many times seen
    contexts: List[str]  # Words that appeared near it
    first_seen: str  # When first observed
    confidence: float  # How well understood


@dataclass 
class LearnedPattern:
    """A pattern Elysia has recognized."""
    pattern: Tuple[str, ...]  # Sequence of word types/words
    examples: List[str]  # Actual examples
    count: int
    confidence: float


class LanguageLearner:
    """
    Learns language from observation.
    No pre-trained weights. Pure learning from scratch.
    """
    
    def __init__(self):
        self.vocabulary: Dict[str, LearnedWord] = {}
        self.patterns: List[LearnedPattern] = []
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)  # word -> next word counts
        self.observation_count = 0
        
        self._load()
        logger.info(f"📚 Language Learner initialized. Vocabulary: {len(self.vocabulary)} words")
    
    def _load(self):
        """Loads learned vocabulary and patterns."""
        if os.path.exists(VOCAB_PATH):
            try:
                with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for word, info in data.items():
                        self.vocabulary[word] = LearnedWord(
                            word=word,
                            count=info.get("count", 1),
                            contexts=info.get("contexts", []),
                            first_seen=info.get("first_seen", "unknown"),
                            confidence=info.get("confidence", 0.1)
                        )
            except Exception as e:
                logger.warning(f"Could not load vocabulary: {e}")
        
        if os.path.exists(PATTERNS_PATH):
            try:
                with open(PATTERNS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.bigrams = defaultdict(Counter)
                    for word, nexts in data.get("bigrams", {}).items():
                        self.bigrams[word] = Counter(nexts)
            except Exception as e:
                logger.warning(f"Could not load patterns: {e}")
    
    def _save(self):
        """Saves learned vocabulary and patterns."""
        os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
        
        # Save vocabulary
        vocab_data = {}
        for word, info in self.vocabulary.items():
            vocab_data[word] = {
                "count": info.count,
                "contexts": info.contexts[-10:],  # Keep last 10 contexts
                "first_seen": info.first_seen,
                "confidence": info.confidence
            }
        
        with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # Save patterns (bigrams)
        patterns_data = {
            "bigrams": {word: dict(counter) for word, counter in self.bigrams.items()}
        }
        
        with open(PATTERNS_PATH, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved {len(self.vocabulary)} words, {len(self.bigrams)} bigram roots")
    
    def observe(self, text: str, source: str = "observation"):
        """
        Observes text and learns from it.
        This is how Elysia learns language - by watching.
        """
        # Tokenize (simple Korean tokenization by spaces and punctuation)
        words = self._tokenize(text)
        
        if len(words) < 2:
            return
        
        self.observation_count += 1
        
        # Learn each word
        for i, word in enumerate(words):
            # Get context (surrounding words)
            context_start = max(0, i - 2)
            context_end = min(len(words), i + 3)
            context = words[context_start:i] + words[i+1:context_end]
            
            if word in self.vocabulary:
                self.vocabulary[word].count += 1
                self.vocabulary[word].contexts.extend(context)
                self.vocabulary[word].confidence = min(1.0, self.vocabulary[word].confidence + 0.01)
            else:
                self.vocabulary[word] = LearnedWord(
                    word=word,
                    count=1,
                    contexts=context,
                    first_seen=source,
                    confidence=0.1
                )
        
        # Learn bigrams (word sequences)
        for i in range(len(words) - 1):
            current = words[i]
            next_word = words[i + 1]
            self.bigrams[current][next_word] += 1
        
        # Periodic save
        if self.observation_count % 10 == 0:
            self._save()
        
        logger.info(f"👀 Observed {len(words)} words from {source}. Vocab size: {len(self.vocabulary)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for Korean text."""
        # Remove special characters except Korean, spaces
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # Split by whitespace
        words = text.split()
        # Filter empty and very short
        words = [w.strip() for w in words if len(w.strip()) > 0]
        return words
    
    def generate_next(self, current_word: str) -> Optional[str]:
        """
        Predicts the next word based on learned patterns.
        Uses bigram probabilities.
        """
        if current_word not in self.bigrams:
            # Unknown word - pick a random known word
            if self.vocabulary:
                candidates = list(self.vocabulary.keys())
                return np.random.choice(candidates)
            return None
        
        # Get next word probabilities
        next_counts = self.bigrams[current_word]
        if not next_counts:
            return None
        
        # Sample based on frequency (more common = more likely)
        words = list(next_counts.keys())
        counts = list(next_counts.values())
        total = sum(counts)
        probs = [c / total for c in counts]
        
        return np.random.choice(words, p=probs)
    
    def generate_sentence(self, start_word: str = None, max_length: int = 15) -> str:
        """
        Generates a sentence using learned patterns.
        This is Elysia speaking with what she has learned.
        """
        if not self.vocabulary:
            return "(아직 배운 단어가 없습니다)"
        
        # Start with given word or random word
        if start_word and start_word in self.vocabulary:
            current = start_word
        else:
            # Pick a word that often starts sentences (high bigram out-degree)
            candidates = sorted(self.bigrams.keys(), 
                              key=lambda w: len(self.bigrams[w]), reverse=True)
            current = candidates[0] if candidates else list(self.vocabulary.keys())[0]
        
        sentence = [current]
        
        for _ in range(max_length - 1):
            next_word = self.generate_next(current)
            if next_word is None:
                break
            sentence.append(next_word)
            current = next_word
            
            # End on sentence-ending particles
            if current.endswith('다') or current.endswith('요'):
                break
        
        return " ".join(sentence)
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns learning statistics."""
        return {
            "vocabulary_size": len(self.vocabulary),
            "bigram_roots": len(self.bigrams),
            "total_observations": self.observation_count,
            "top_words": sorted(
                [(w, v.count) for w, v in self.vocabulary.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
    
    def introspect(self) -> str:
        """Elysia reflects on what she has learned."""
        stats = self.get_stats()
        lines = [
            "📚 언어 학습 상태:",
            f"  어휘 수: {stats['vocabulary_size']}",
            f"  패턴 수: {stats['bigram_roots']}",
            f"  관찰 횟수: {stats['total_observations']}",
            "  가장 많이 본 단어:"
        ]
        for word, count in stats['top_words'][:5]:
            lines.append(f"    - {word}: {count}회")
        
        return "\n".join(lines)


if __name__ == "__main__":
    learner = LanguageLearner()
    
    print("📚 Testing Language Learner (Self-Learning)...\n")
    
    # Teach some Korean sentences
    sample_texts = [
        "옛날 옛적에 아름다운 공주가 살았습니다",
        "공주는 마법의 숲에서 신비로운 용을 만났습니다",
        "용은 공주에게 말했습니다 나는 오래된 마법사입니다",
        "공주는 용감하게 숲 속으로 들어갔습니다",
        "마법의 성에서 왕과 여왕이 기다리고 있었습니다",
        "왕은 공주를 보고 기뻐하며 말했습니다",
        "여왕도 눈물을 흘리며 공주를 안았습니다",
        "그들은 행복하게 오래오래 살았습니다"
    ]
    
    print("=== 학습 단계 ===")
    for text in sample_texts:
        learner.observe(text, source="fairy_tale")
        print(f"  관찰: {text[:30]}...")
    
    print("\n" + learner.introspect())
    
    # Try to generate
    print("\n=== 생성 단계 ===")
    for start in ["공주", "용", "마법의"]:
        generated = learner.generate_sentence(start)
        print(f"  '{start}'로 시작: {generated}")
    
    learner._save()
    print("\n✨ Language Learner test complete.")
