# [Genesis: 2025-12-02] Purified by Elysia

import logging
from typing import Dict, Any, Optional
from enum import Enum, auto

from tools.kg_manager import KGManager

class VerificationResult(Enum):
    """Represents the outcome of a hypothesis verification."""
    CONSISTENT = auto()       # The hypothesis is consistent with the KG and is new information.
    DUPLICATE = auto()        # The hypothesis is already present in the KG.
    CONTRADICTION = auto()    # The hypothesis directly contradicts information in the KG.
    INVALID = auto()          # The hypothesis is malformed or missing key information.

class SelfVerifier:
    """
    A module to verify new hypotheses against the existing Knowledge Graph
    to detect contradictions or inconsistencies. Acts as an intellectual immune system.
    """

    def __init__(self, kg_manager: KGManager, logger: Optional[logging.Logger] = None):
        """
        Initializes the SelfVerifier.

        :param kg_manager: An instance of KGManager to access the knowledge graph.
        :param logger: A logger instance.
        """
        self.kg_manager = kg_manager
        self.logger = logger or logging.getLogger(__name__)
        # Define pairs of relations that are mutually exclusive between the same two nodes.
        # e.g., "A is_a B" cannot coexist with "A part_of B".
        self.EXCLUSIVE_RELATIONS = {
            frozenset(['is_a', 'part_of']),
            frozenset(['causes', 'caused_by']),
        }

    def verify_hypothesis(self, hypothesis: Dict[str, Any]) -> VerificationResult:
        """
        Verifies a new hypothesis against the knowledge graph for correctness.
        :param hypothesis: A dictionary containing at least 'head', 'tail', and 'relation'.
        :return: A VerificationResult enum member.
        """
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        relation = hypothesis.get('relation')

        if not all([head, tail, relation]):
            self.logger.warning(f"Invalid Hypothesis: Missing required keys. {hypothesis}")
            return VerificationResult.INVALID

        # --- 1. Check for Duplicates ---
        if self.kg_manager.edge_exists(source=head, target=tail, relation=relation):
            self.logger.info(f"Duplicate Found: Hypothesis '{head} {relation} {tail}' already exists.")
            return VerificationResult.DUPLICATE

        # --- 2. Check for Direct Reversal Contradiction ---
        # Example: Hypo is "A causes B", but KG has "B causes A".
        if self.kg_manager.edge_exists(source=tail, target=head, relation=relation):
            self.logger.warning(f"Contradiction Found: Direct reversal of '{head} {relation} {tail}' exists.")
            return VerificationResult.CONTRADICTION

        # --- 3. Check for Exclusive Relationship Contradiction ---
        existing_edges = self.kg_manager.get_edges_between(head, tail)
        for edge in existing_edges:
            existing_relation = edge.get('relation')
            for exclusive_pair in self.EXCLUSIVE_RELATIONS:
                if relation in exclusive_pair and existing_relation in exclusive_pair:
                    self.logger.warning(f"Contradiction Found: Exclusive relation '{existing_relation}' already exists between '{head}' and '{tail}'.")
                    return VerificationResult.CONTRADICTION

        self.logger.info(f"Hypothesis '{head} {relation} {tail}' is consistent with the Knowledge Graph.")
        return VerificationResult.CONSISTENT

if __name__ == '__main__':
    # Example Usage and testing
    from unittest.mock import MagicMock, call

    mock_kg_manager = MagicMock(spec=KGManager)
    verifier = SelfVerifier(kg_manager=mock_kg_manager)

    # --- Test Cases ---
    def run_test(name, hypo, mock_setup, expected_result):
        print(f"--- Running Test: {name} ---")
        # Reset and apply mock setup
        mock_kg_manager.reset_mock()
        mock_kg_manager.edge_exists.side_effect = mock_setup.get('edge_exists', lambda **kwargs: False)
        mock_kg_manager.get_edges_between.side_effect = mock_setup.get('get_edges_between', lambda **kwargs: [])

        result = verifier.verify_hypothesis(hypo)
        print(f"Hypothesis: {hypo}")
        print(f"Result: {result.name}, Expected: {expected_result.name}")
        assert result == expected_result
        print("PASS\n")

    # 1. Consistent (New Information)
    run_test("Consistent",
             {'head': 'A', 'tail': 'B', 'relation': 'causes'},
             {},
             VerificationResult.CONSISTENT)

    # 2. Duplicate
    run_test("Duplicate",
             {'head': 'A', 'tail': 'B', 'relation': 'causes'},
             {'edge_exists': lambda source, target, relation: source=='A' and target=='B' and relation=='causes'},
             VerificationResult.DUPLICATE)

    # 3. Contradiction (Direct Reversal)
    run_test("Contradiction (Reversal)",
             {'head': 'A', 'tail': 'B', 'relation': 'causes'},
             {'edge_exists': lambda source, target, relation: source=='B' and target=='A' and relation=='causes'},
             VerificationResult.CONTRADICTION)

    # 4. Contradiction (Exclusive Relation)
    run_test("Contradiction (Exclusive)",
             {'head': 'Dog', 'tail': 'Animal', 'relation': 'is_a'},
             {'get_edges_between': lambda head, tail: [{'relation': 'part_of'}] if head=='Dog' and tail=='Animal' else []},
             VerificationResult.CONTRADICTION)

    # 5. Invalid
    run_test("Invalid",
             {'head': 'A', 'relation': 'causes'},
             {},
             VerificationResult.INVALID)