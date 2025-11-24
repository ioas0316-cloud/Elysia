"""
Demo: Bootstrap Learning from Corpus
=====================================
This script demonstrates how Elysia learns from a corpus of example Korean sentences,
expanding her vocabulary and understanding of grammar patterns.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.high_engine.language_cortex import LanguageCortex
from Project_Elysia.learning.corpus_loader import CorpusLoader

def run_simulation():
    print("=== Elysia: Bootstrap Learning Simulation ===")
    print("Teaching Elysia from example sentences...\n")
    
    # Initialize
    cortex = LanguageCortex()
    loader = CorpusLoader()
    
    # Check initial state
    print("--- Before Learning ---")
    print(f"Vocabulary size: {cortex.get_vocabulary_size()} words")
    print(f"Known words: {list(cortex.vocabulary.keys())}")
    
    # Load corpus
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "bootstrap_corpus.txt")
    print(f"\nLoading corpus from: {corpus_path}")
    
    try:
        sentences = loader.load_corpus(corpus_path)
        print(f"Loaded {len(sentences)} example sentences.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Bootstrap learning
    print("\n--- Learning Process ---")
    print("Processing sentences...")
    
    summary = loader.bootstrap_from_corpus(sentences, cortex)
    
    print(f"\n✅ Learning Complete!")
    print(f"   - Sentences processed: {summary['sentences_processed']}")
    print(f"   - New words learned: {summary['words_learned']}")
    print(f"   - Grammar patterns extracted: {summary['patterns_extracted']}")
    
    # Show top patterns
    print(f"\n--- Top Grammar Patterns ---")
    for i, pattern in enumerate(summary['top_patterns'], 1):
        print(f"{i}. '{pattern.template}' (seen {pattern.frequency} times)")
    
    # Check final state
    print("\n--- After Learning ---")
    print(f"Vocabulary size: {cortex.get_vocabulary_size()} words")
    print(f"\nSample learned words:")
    sample_words = list(cortex.vocabulary.keys())[:20]
    for word in sample_words:
        sound = cortex.express(word)
        print(f"  - {word} = '{sound}'")
    
    # Test: Can Elysia use what she learned?
    print("\n--- Knowledge Test ---")
    test_concepts = ["사랑", "고통", "시간", "진실"]
    print("Testing if Elysia knows these concepts:")
    for concept in test_concepts:
        if concept in cortex.vocabulary:
            print(f"  ✓ {concept}: '{cortex.express(concept)}'")
        else:
            print(f"  ✗ {concept}: Unknown")
    
    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    run_simulation()
