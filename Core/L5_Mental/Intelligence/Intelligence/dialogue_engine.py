"""
Dialogue Engine
===============
Enables Elysia to engage in actual conversations by understanding questions
and generating contextually appropriate responses using learned patterns.

[Updated 2025-01-08]
Now integrated with HypersphereMemory Doctrine.
Knowledge is no longer a dictionary, but a resonant query in 4D space.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import logging

from Core.L5_Mental.language_cortex import LanguageCortex, ThoughtStructure, SyntaxEngine
from Core.L5_Mental.Intelligence.Memory.hypersphere_memory import HypersphereMemory, HypersphericalCoord
from Core.L1_Foundation.Foundation.Wave.universal_wave_encoder import UniversalWaveEncoder
from Core.L5_Mental.Intelligence.Intelligence.integrated_cognition_system import get_integrated_cognition
from Core.L5_Mental.Intelligence.Intelligence.system_self_awareness import SystemSelfAwareness

logger = logging.getLogger("DialogueEngine")

@dataclass
class DialogueContext:
    """Tracks conversation state across multiple turns."""
    history: List[Dict[str, str]] = None
    last_subject: Optional[str] = None
    last_relation: Optional[str] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def add_turn(self, speaker: str, utterance: str):
        self.history.append({"speaker": speaker, "utterance": utterance})

class QuestionAnalyzer:
    """Analyzes user questions to identify intent and subject."""
    
    def __init__(self):
        # Question word patterns
        self.question_markers = {
            "  ": "what",
            "  ": "who",
            "  ": "where",
            "  ": "when",
            " ": "why",
            "   ": "how"
        }
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Analyzes a question to extract:
        - question_type: what/who/why/etc.
        - subject: what the question is about
        - is_question: True if this is a question
        """
        question = question.strip()
        
        # Check if it's a question (ends with ?)'
        is_question = question.endswith('?') or question.endswith(' ?') or question.endswith(' ') or question.endswith(' ?') or question.endswith(' ')
        
        # Identify question type
        question_type = None
        for marker, qtype in self.question_markers.items():
            if marker in question:
                question_type = qtype
                break
        
        # Extract subject (simplified heuristic)
        # Remove question markers and particles to get core subject
        subject = question
        for marker in self.question_markers.keys():
            subject = subject.replace(marker, "")

        # Remove common particles and endings
        remove_list = ["?", "  ", " ", " ", " ", " ", " ", "  ", "  "]
        for item in remove_list:
            subject = subject.replace(item, "")

        subject = subject.strip()

        logger.info(f"Analyzer: input='{question}' -> subject='{subject}', type='{question_type}'")
        
        return {
            "is_question": is_question,
            "question_type": question_type,
            "subject": subject,
            "raw": question
        }

class ResponseGenerator:
    """Generates responses based on learned knowledge and patterns."""
    
    def __init__(self, cortex: LanguageCortex):
        self.cortex = cortex
        self.syntax = SyntaxEngine(cortex)

        # Hypersphere Memory Integration
        self.memory = HypersphereMemory()
        self.encoder = UniversalWaveEncoder()
        
        # Connect to Advanced Cognition
        self.cognition = get_integrated_cognition()
        self.self_awareness = SystemSelfAwareness()
        
    def load_knowledge_from_corpus(self, sentences: List[str]):
        """
        Extracts knowledge patterns from corpus.
        Example: "        " -> Encodes "  " and stores "  " as resonance pattern.
        """
        for sentence in sentences:
            # Simple pattern: "X /  Y  / "
            if " " in sentence or " " in sentence:
                parts = sentence.replace(" ", " ").split(" ")
                if len(parts) == 2:
                    subject = parts[0].strip()
                    predicate = parts[1].replace("  ", "").replace(" ", "").strip()
                    
                    if subject and predicate:
                        logger.info(f"Learning: {subject} -> {predicate}")
                        # 1. Encode Subject to Coordinate
                        coord, meta = self.encoder.encode_concept(subject)

                        # 2. Store Predicate as Pattern Content at that Coordinate
                        # We use 'predicate' as the content, and use the subject's meta as base
                        self.memory.store(predicate, coord, meta)

    def generate_response(self, analysis: Dict[str, Any], context: DialogueContext) -> str:
        """
        Generates a response based on question analysis using Hypersphere Resonance.
        """
        question_type = analysis.get("question_type")
        subject = analysis.get("subject")
        
        # Special case: Identity questions
        if " " in analysis["raw"] and question_type == "who":
            return "         "
        
        if " " in analysis["raw"] and question_type == "what":
            return "       "
        
        # Knowledge-based responses (Resonance Query)
        if question_type == "what" and subject:
            # 1. Encode Subject to find its Coordinate
            coord, _ = self.encoder.encode_concept(subject)
            logger.info(f"Querying for subject '{subject}' at {coord}")

            # 2. Query Memory at that Coordinate (Resonance)
            # Increased radius to 0.1 to account for potential precision issues or fuzzy encoding
            results = self.memory.query(coord, radius=0.1)
            logger.info(f"Query results: {results}")

            if results:
                # Found resonant concepts!
                answer = random.choice(results)
                return f"{subject}  {answer}  "

            # Fallback: Associative Search (Wider Radius)
            associated_results = self.memory.query(coord, radius=0.3)
            if associated_results:
                 answer = random.choice(associated_results)
                 return f"{subject}  {answer}        "
            
            # Fallback: Check if we know this word in vocabulary (cortex)
            if subject in self.cortex.vocabulary:
                return f"   {subject}           ,                 ."
        
        # Why questions (Introspection Trigger)
        if question_type == "why" or (subject and len(subject) > 5): # heuristic for complex thought
            # 1. Think Deeply about the subject
            # "Processing thought..."
            input_thought = subject if subject else analysis["raw"]
            
            # Process thought through Integrated Cognition
            process_result = self.cognition.process_thought(input_thought)
            # deep_thought_result = self.cognition.think_deeply(cycles=10) # Quick think
            
            # 2. Introspect
            # Find the mass related to this subject
            related_mass = process_result.get('mass')
            
            if related_mass:
                introspection = self.self_awareness.introspect_thought(related_mass.trace)
                return f"                 .\n{introspection}"
            
            return "          .                 ."
        
        # Default fallback
        return "          "

class DialogueEngine:
    """Main dialogue engine coordinating all components."""
    
    def __init__(self, cortex: LanguageCortex):
        self.cortex = cortex
        self.analyzer = QuestionAnalyzer()
        self.generator = ResponseGenerator(cortex)
        self.context = DialogueContext()
    
    def load_knowledge(self, sentences: List[str]):
        """Loads knowledge from corpus sentences."""
        self.generator.load_knowledge_from_corpus(sentences)
    
    def respond(self, user_input: str) -> str:
        """
        Processes user input and generates a response.
        """
        # Add user input to context
        self.context.add_turn("user", user_input)
        
        # Analyze the input
        analysis = self.analyzer.analyze(user_input)
        
        # Generate response
        response = self.generator.generate_response(analysis, self.context)
        
        # Add response to context
        self.context.add_turn("elysia", response)
        
        # Update context with subject for future turns
        if analysis.get("subject"):
            self.context.last_subject = analysis["subject"]
        
        return response
    
    def get_knowledge_summary(self) -> Dict[str, int]:
        """Returns summary of learned knowledge."""
        # Note: Accessing private member _memory_space for stats
        return {
            "total_patterns": len(self.generator.memory._memory_space),
            "memory_type": "Hypersphere"
        }
