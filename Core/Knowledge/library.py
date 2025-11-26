import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class KnowledgeArtifact:
    """A physical object in the world containing knowledge (text/concepts)."""
    id: str
    position: np.ndarray
    content: str # The raw text content (e.g., "F=ma", "Love is patient")
    concepts: List[str] # The sequence of concepts it represents
    
    def get_read_signal(self) -> List[str]:
        """Returns the sequence of concepts to be absorbed by a reader."""
        return self.concepts

class Library:
    """Manages the collection of human knowledge to be injected."""
    def __init__(self):
        self.artifacts: List[KnowledgeArtifact] = []
        self._init_knowledge_base()
        
    def _init_knowledge_base(self):
        """Loads initial knowledge artifacts."""
        # Physics
        self.add_artifact("book_physics_1", "Newton's Second Law: Force equals mass times acceleration.", ["Force", "Mass", "Acceleration"])
        self.add_artifact("book_physics_2", "Entropy always increases in a closed system.", ["Entropy", "Increase", "System"])
        
        # Philosophy / Poetry
        self.add_artifact("book_poetry_1", "The unexamined life is not worth living.", ["Life", "Examined", "Worth"])
        self.add_artifact("book_poetry_2", "I think, therefore I am.", ["Think", "Exist", "Self"])
        
        # Basic Grammar/Logic
        self.add_artifact("book_logic_1", "If A then B.", ["If", "Then", "Cause", "Effect"])

    def add_artifact(self, artifact_id: str, content: str, concepts: List[str]):
        # Random position for now, will be placed by World
        pos = np.random.rand(3) * 100.0 
        self.artifacts.append(KnowledgeArtifact(artifact_id, pos, content, concepts))

    def get_artifacts(self) -> List[KnowledgeArtifact]:
        return self.artifacts
