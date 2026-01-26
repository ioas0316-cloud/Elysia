"""
Domain Absorber (       )
==============================
Core.L5_Mental.Induction.domain_absorber

"Any domain is a force vector. I absorb them all."

This module enables Elysia to absorb knowledge from any domain
(code, math, language, art) and convert it into 7D Qualia force vectors.
"""

import logging
import json
import os
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

logger = logging.getLogger("Elysia.Induction")


@dataclass
class DomainQualia:
    """A domain represented as a 7D Qualia vector."""
    name: str
    description: str
    qualia_vector: List[float]  # 7 dimensions
    source_type: str  # "text", "code", "model", "experience"
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self.qualia_vector, dtype=np.float32)


# The 7 Qualia Dimensions
QUALIA_DIMENSIONS = [
    "LOGIC",      #        
    "CREATIVITY", #        
    "PRECISION",  #        
    "ABSTRACTION", #       
    "EMOTION",    #       
    "UTILITY",    #       
    "MYSTERY"     #     /      
]


class DomainAbsorber:
    """
    The Universal Domain Induction Engine.
    Converts any knowledge domain into 7D Qualia space.
    """
    
    def __init__(self, storage_path: str = "data/Qualia/domains.json"):
        self.storage_path = storage_path
        self.domains: Dict[str, DomainQualia] = {}
        self._load()
        
        logger.info(f"  Domain Absorber initialized. {len(self.domains)} domains loaded.")
    
    def _load(self):
        """Loads absorbed domains from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, info in data.items():
                        self.domains[name] = DomainQualia(
                            name=name,
                            description=info.get("description", ""),
                            qualia_vector=info.get("qualia_vector", [0.5]*7),
                            source_type=info.get("source_type", "unknown"),
                            confidence=info.get("confidence", 0.5),
                            metadata=info.get("metadata", {})
                        )
            except Exception as e:
                logger.warning(f"   Could not load domains: {e}")
    
    def _save(self):
        """Saves absorbed domains to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        data = {}
        for name, domain in self.domains.items():
            data[name] = asdict(domain)
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Saved {len(self.domains)} domains.")
    
    def absorb_text(self, name: str, text: str, description: str = "") -> DomainQualia:
        """
        Absorbs a text-based domain (e.g., a concept, a field of knowledge).
        """
        # Generate Qualia vector from text characteristics
        qualia = self._text_to_qualia(text)
        
        domain = DomainQualia(
            name=name,
            description=description or f"Domain absorbed from text: {name}",
            qualia_vector=qualia,
            source_type="text",
            confidence=0.7,
            metadata={"text_length": len(text), "hash": hashlib.md5(text.encode()).hexdigest()[:8]}
        )
        
        self.domains[name] = domain
        self._save()
        
        logger.info(f"  Absorbed text domain: {name}")
        return domain
    
    def absorb_code(self, name: str, code: str, language: str = "python") -> DomainQualia:
        """
        Absorbs a code-based domain.
        """
        qualia = self._code_to_qualia(code, language)
        
        domain = DomainQualia(
            name=name,
            description=f"Code domain ({language}): {name}",
            qualia_vector=qualia,
            source_type="code",
            confidence=0.8,
            metadata={"language": language, "lines": code.count('\n') + 1}
        )
        
        self.domains[name] = domain
        self._save()
        
        logger.info(f"  Absorbed code domain: {name}")
        return domain
    
    def absorb_principle(self, name: str, principle: Dict[str, float]) -> DomainQualia:
        """
        Directly absorbs a principle as explicit Qualia values.
        
        Args:
            name: Domain name
            principle: Dict with keys from QUALIA_DIMENSIONS, values 0.0-1.0
        """
        qualia = [principle.get(dim, 0.5) for dim in QUALIA_DIMENSIONS]
        
        domain = DomainQualia(
            name=name,
            description=f"Principle-based domain: {name}",
            qualia_vector=qualia,
            source_type="principle",
            confidence=1.0,
            metadata={"explicit": True}
        )
        
        self.domains[name] = domain
        self._save()
        
        logger.info(f"  Absorbed principle domain: {name}")
        return domain
    
    def _text_to_qualia(self, text: str) -> List[float]:
        """Converts text characteristics to 7D Qualia."""
        text_lower = text.lower()
        
        # Heuristic extraction (can be replaced with embeddings later)
        logic = 0.5 + 0.3 * (text_lower.count("therefore") + text_lower.count("because")) / max(1, len(text)/100)
        creativity = 0.5 + 0.3 * (text_lower.count("imagine") + text_lower.count("create")) / max(1, len(text)/100)
        precision = 0.5 + 0.3 * (text.count("=") + text.count("%") + text.count("exactly")) / max(1, len(text)/100)
        abstraction = 0.5 + 0.3 * (text_lower.count("concept") + text_lower.count("principle")) / max(1, len(text)/100)
        emotion = 0.5 + 0.3 * (text_lower.count("feel") + text_lower.count("love") + text_lower.count("fear")) / max(1, len(text)/100)
        utility = 0.5 + 0.3 * (text_lower.count("use") + text_lower.count("apply") + text_lower.count("practical")) / max(1, len(text)/100)
        mystery = 0.5 + 0.3 * (text_lower.count("unknown") + text_lower.count("mystery") + text_lower.count("?")) / max(1, len(text)/100)
        
        return [min(1.0, max(0.0, v)) for v in [logic, creativity, precision, abstraction, emotion, utility, mystery]]
    
    def _code_to_qualia(self, code: str, language: str) -> List[float]:
        """Converts code characteristics to 7D Qualia."""
        # Code tends to be high in LOGIC, PRECISION, UTILITY
        lines = code.count('\n') + 1
        functions = code.count('def ') + code.count('function ')
        classes = code.count('class ')
        comments = code.count('#') + code.count('//')
        
        logic = 0.8  # Code is inherently logical
        creativity = min(1.0, 0.3 + 0.1 * classes)  # More classes = more abstraction/creativity
        precision = min(1.0, 0.7 + 0.02 * (code.count('==') + code.count('!=')))
        abstraction = min(1.0, 0.4 + 0.05 * (classes + functions))
        emotion = min(0.5, 0.1 + 0.02 * comments)  # Comments can express intent
        utility = 0.9  # Code is practical
        mystery = max(0.0, 0.3 - 0.01 * comments)  # More comments = less mystery
        
        return [logic, creativity, precision, abstraction, emotion, utility, mystery]
    
    def get_domain(self, name: str) -> Optional[DomainQualia]:
        """Retrieves an absorbed domain."""
        return self.domains.get(name)
    
    def find_similar(self, query_qualia: List[float], top_k: int = 3) -> List[tuple]:
        """
        Finds domains most similar to a query Qualia vector.
        Returns list of (name, similarity_score) tuples.
        """
        query = np.array(query_qualia)
        results = []
        
        for name, domain in self.domains.items():
            domain_vec = domain.to_numpy()
            # Cosine similarity
            similarity = np.dot(query, domain_vec) / (np.linalg.norm(query) * np.linalg.norm(domain_vec) + 1e-8)
            results.append((name, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def list_domains(self) -> List[str]:
        """Lists all absorbed domain names."""
        return list(self.domains.keys())


if __name__ == "__main__":
    absorber = DomainAbsorber()
    
    print("  Testing Domain Absorber...\n")
    
    # Absorb some test domains
    absorber.absorb_principle("MATHEMATICS", {
        "LOGIC": 0.95,
        "CREATIVITY": 0.6,
        "PRECISION": 0.99,
        "ABSTRACTION": 0.9,
        "EMOTION": 0.1,
        "UTILITY": 0.8,
        "MYSTERY": 0.3
    })
    
    absorber.absorb_principle("POETRY", {
        "LOGIC": 0.3,
        "CREATIVITY": 0.95,
        "PRECISION": 0.4,
        "ABSTRACTION": 0.7,
        "EMOTION": 0.95,
        "UTILITY": 0.2,
        "MYSTERY": 0.8
    })
    
    absorber.absorb_principle("PROGRAMMING", {
        "LOGIC": 0.9,
        "CREATIVITY": 0.6,
        "PRECISION": 0.95,
        "ABSTRACTION": 0.7,
        "EMOTION": 0.2,
        "UTILITY": 0.99,
        "MYSTERY": 0.1
    })
    
    print(f"Domains absorbed: {absorber.list_domains()}")
    
    # Test similarity search
    query = [0.9, 0.5, 0.9, 0.8, 0.1, 0.9, 0.2]  # Similar to programming
    similar = absorber.find_similar(query)
    print(f"\nMost similar to query: {similar}")
    
    print("\n  Domain Absorber test complete.")
