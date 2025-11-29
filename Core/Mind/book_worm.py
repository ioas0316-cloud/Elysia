
import logging
import re
import math
from typing import List, Dict, Any
from collections import Counter

from Core.Mind.hippocampus import Hippocampus

logger = logging.getLogger("BookWorm")

class BookWorm:
    """
    The Librarian of Elysia.
    Ingests raw text (books), extracts concepts, and analyzes style.
    """
    def __init__(self, hippocampus: Hippocampus):
        self.hippocampus = hippocampus
        
    def ingest_book(self, title: str, author: str, text: str):
        """
        Ingest a full book.
        """
        logger.info(f"📚 BookWorm: Ingesting '{title}' by {author}...")
        
        # 1. Create Book Concept
        book_id = f"Book: {title}"
        self.hippocampus.add_concept(
            book_id, 
            concept_type="source", 
            metadata={"author": author, "type": "book"}
        )
        
        # 2. Chunking (Paragraphs)
        chunks = self._chunk_text(text)
        logger.info(f"   -> Split into {len(chunks)} chunks.")
        
        # 3. Process Chunks
        for i, chunk in enumerate(chunks):
            # Extract Concepts (Keywords)
            concepts = self._extract_concepts(chunk)
            
            # Analyze Style (Sentiment/Abstraction)
            style_vec = self._analyze_style(chunk)
            
            # Ingest Concepts
            for concept, count in concepts.items():
                # Add concept with style vector
                # We blend the book's style into the concept's vector
                self.hippocampus.add_concept(
                    concept, 
                    concept_type="learned_concept",
                    metadata={"source": book_id, "frequency": count}
                )
                
                # Link to Book
                self.hippocampus.add_causal_link(book_id, concept, weight=0.1 * count)
                
                # TODO: Update concept vector based on style_vec
                # This requires a method in Hippocampus or direct access to ResonanceEngine
                # For now, we assume add_concept handles basic initialization.
                
            if i % 10 == 0:
                logger.info(f"   -> Processed chunk {i}/{len(chunks)}")
                
        logger.info(f"✅ Finished ingesting '{title}'.")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Simple split by double newline
        chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        return chunks

    def _extract_concepts(self, text: str) -> Dict[str, int]:
        """
        Extract interesting words (Nouns/Verbs).
        Simple regex-based extraction for now.
        """
        # Remove punctuation
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        # Filter stopwords (very basic list)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "for", "is", "was", "it", "he", "she", "they", "that", "this", "with", "as"}
        
        interesting_words = [w for w in words if w not in stopwords and len(w) > 3]
        
        return Counter(interesting_words)

    def _analyze_style(self, text: str) -> List[float]:
        """
        Analyze text style to generate a 3D 'Will Vector'.
        x: Logic vs Emotion (0.0 to 1.0)
        y: Positivity vs Negativity (0.0 to 1.0)
        z: Concrete vs Abstract (0.0 to 1.0)
        """
        # 1. Logic vs Emotion (x)
        # Check for logical connectors vs emotional words
        logical_words = {"therefore", "because", "if", "then", "logic", "reason", "fact"}
        emotional_words = {"feel", "love", "hate", "sad", "happy", "joy", "pain", "heart"}
        
        l_count = sum(1 for w in text.lower().split() if w in logical_words)
        e_count = sum(1 for w in text.lower().split() if w in emotional_words)
        
        total = l_count + e_count
        if total == 0:
            x = 0.5
        else:
            x = e_count / total # 1.0 = Pure Emotion, 0.0 = Pure Logic
            
        # 2. Positivity (y)
        # Very simple sentiment
        pos_words = {"good", "great", "light", "hope", "love", "yes", "beautiful"}
        neg_words = {"bad", "dark", "fear", "pain", "no", "ugly", "death"}
        
        p_count = sum(1 for w in text.lower().split() if w in pos_words)
        n_count = sum(1 for w in text.lower().split() if w in neg_words)
        
        total_s = p_count + n_count
        if total_s == 0:
            y = 0.5
        else:
            y = p_count / total_s
            
        # 3. Abstract (z)
        # Average word length as a proxy for complexity/abstraction?
        words = text.split()
        if not words:
            z = 0.5
        else:
            avg_len = sum(len(w) for w in words) / len(words)
            # Normalize: avg length 4 -> 0.0, avg length 8 -> 1.0
            z = min(max((avg_len - 4) / 4, 0.0), 1.0)
            
        return [x, y, z]
