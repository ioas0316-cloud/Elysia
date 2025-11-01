"""
Self-Reflection Cortex for Elysia

This module enables Elysia to introspect her own knowledge graph,
identify contradictions, uncertainties, and areas for potential growth.
"""
import re
from typing import Dict, List, Optional, Tuple
from tools.kg_manager import KGManager

class SelfReflectionCortex:
    def __init__(self, kg_manager: KGManager):
        """
        Initializes the cortex with a KGManager instance.
        """
        self.kg_manager = kg_manager

    def analyze_input(self, text_input: str) -> Optional[Dict]:
        """
        Analyzes a new piece of information and checks for contradictions
        with the existing knowledge graph.

        Returns a dictionary describing the conflict if one is found, otherwise None.
        """
        # This is a placeholder for a sophisticated NLP parsing logic.
        # For now, we'll use a simple pattern: "Subject is/is_a/causes Not a/an/the Predicate"
        # Example: "Socrates is not a human"

        # Simple parsing logic for "A is not B" type contradictions.
        # Format: (Subject, Relation, Object, is_negated)
        parsed_statement = self._parse_statement(text_input)

        if parsed_statement:
            subject, relation, obj, is_negated = parsed_statement

            # If the statement is negative (e.g., "Socrates is NOT a Human")
            if is_negated:
                # Check if a positive version exists in the KG
                existing_edge = self.kg_manager.get_edge(subject, obj, relation)
                if existing_edge:
                    return {
                        "type": "contradiction",
                        "statement": text_input,
                        "conflicting_knowledge": existing_edge,
                        "proposed_action": {
                            "type": "remove_edge",
                            "source": subject,
                            "target": obj,
                            "relation": relation
                        }
                    }

        return None

    def _parse_statement(self, text: str) -> Optional[Tuple[str, str, str, bool]]:
        """
        A very basic NLP parser to extract a simple statement.
        This should be replaced with a more robust parser in the future.

        Example patterns:
        - "Socrates is a human" -> ("socrates", "is_a", "human", False)
        - "Socrates is not a human" -> ("socrates", "is_a", "human", True)
        """
        text = text.lower().strip()

        # Pattern 1: "is not a" / "is a"
        match = re.search(r"(.+?)\s+is\s+(not\s+)?a(?:n)?\s+(.+)", text)
        if match:
            subject = match.group(1).strip()
            is_negated = bool(match.group(2))
            obj = match.group(3).strip()
            return (subject, "is_a", obj, is_negated)

        return None
