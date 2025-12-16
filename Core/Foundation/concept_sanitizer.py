"""
Concept Sanitizer ( The Kidney )
================================
"To purify is to define suitable boundaries."

This module is the gatekeeper for the TorchGraph. 
It filters out "Digital Noise" (e.g., 'Star -175283925', 'Object_0x123', 'None').

Rules:
1.  No pure numbers or ID-like patterns.
2.  No "programmatic artifacts" (null, undefined, object).
3.  Must be semantically meaningful (Subjective, but heuristic-based).
"""

import re
import logging

logger = logging.getLogger("ConceptSanitizer")

class ConceptSanitizer:
    def __init__(self):
        # Regex for "Trash"
        self.trash_patterns = [
            r"^\d+$",               # Pure numbers
            r"-\d+",                # Negative numbers / IDs
            r"0x[a-fA-F0-9]+",      # Hex codes
            r"Object_\d+",          # Default object names
            r"Star [-]?\d+",        # Specific user complaint
            r"[a-f0-9]{8}-",        # UUID fragments
            r"undefined", r"null", r"None" # Nulls
        ]
        
    def is_valid(self, concept: str) -> bool:
        """
        Returns True if the concept is worthy of the Crystal.
        """
        if not concept or not isinstance(concept, str): return False
        
        concept = concept.strip()
        
        # 1. Length Check
        if len(concept) < 2: return False # Single chars are rarely concepts (except 'I', 'A')
        if len(concept) > 50: return False # Too long = Description, not Concept
        
        # 2. Trash Pattern Check
        for pattern in self.trash_patterns:
            if re.search(pattern, concept, re.IGNORECASE):
                logger.warning(f"🛡️  Blocked Trash Concept: '{concept}'")
                return False
                
        # 3. Charset Check (Allow letters, spaces, hyphens. Disallow heavy symbols)
        if re.search(r"[<>{}\[\]@#$%^&*]", concept): # Code syntax
            return False
            
        return True

    def sanitize(self, concept: str) -> str:
        """
        Cleans a concept string (e.g. " The  Cat " -> "Cat").
        """
        if not concept: return ""
        cleaned = concept.strip().title() # Normalize casing
        
        # Remove trailing punctuation
        cleaned = re.sub(r"[.,;:]$", "", cleaned)
        
        return cleaned

# Singleton
_sanitizer = None
def get_sanitizer():
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = ConceptSanitizer()
    return _sanitizer
