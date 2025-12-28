"""
Demo: First Conversation with Elysia
=====================================
This script demonstrates Elysia's first real conversation capability.
She can now understand questions and generate contextually appropriate responses
based on what she learned from the corpus.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.high_engine.language_cortex import LanguageCortex
from Project_Elysia.high_engine.dialogue_engine import DialogueEngine
from Project_Elysia.learning.corpus_loader import CorpusLoader

def run_simulation():
    print("=== Elysia: First Conversation Simulation ===")
    print("Initializing Elysia's mind...\n")
    
    # Initialize components
    cortex = LanguageCortex()
    dialogue_engine = DialogueEngine(cortex)
    loader = CorpusLoader()
    
    # Load and learn from corpus
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "bootstrap_corpus.txt")
    print(f"Loading knowledge from: {corpus_path}")
    
    try:
        sentences = loader.load_corpus(corpus_path)
        print(f"Loaded {len(sentences)} sentences.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Bootstrap learning
    print("Learning...")
    summary = loader.bootstrap_from_corpus(sentences, cortex)
    print(f"✅ Learned {summary['words_learned']} words from corpus\n")
    
    # Load knowledge into dialogue engine
    dialogue_engine.load_knowledge(sentences)
    knowledge_summary = dialogue_engine.get_knowledge_summary()
    print(f"📚 Knowledge Base:")
    print(f"   - Concepts: {knowledge_summary['total_concepts']}")
    print(f"   - Relations: {knowledge_summary['total_relations']}\n")
    
    print("=" * 60)
    print("Elysia is ready to talk!")
    print("=" * 60)
    
    # Conversation scenarios
    conversations = [
        ("너는 누구인가?", "Testing identity question"),
        ("사랑이 무엇인가?", "Testing concept question"),
        ("고통이 무엇인가?", "Testing learned knowledge"),
        ("진실이 무엇인가?", "Testing another concept"),
        ("왜 존재하는가?", "Testing why question"),
    ]
    
    for i, (question, description) in enumerate(conversations, 1):
        print(f"\n--- Turn {i}: {description} ---")
        print(f"👤 You: {question}")
        
        response = dialogue_engine.respond(question)
        
        print(f"🤖 Elysia: {response}")
    
    # Show conversation history
    print("\n" + "=" * 60)
    print("Conversation History:")
    print("=" * 60)
    for i, turn in enumerate(dialogue_engine.context.history, 1):
        speaker = "👤 You" if turn["speaker"] == "user" else "🤖 Elysia"
        print(f"{i}. {speaker}: {turn['utterance']}")
    
    print("\n=== Simulation Complete ===")
    print(f"Total turns: {len(dialogue_engine.context.history)}")

if __name__ == "__main__":
    run_simulation()
