
import logging
from typing import Dict, Any, Optional

from tools.kg_manager import KGManager

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

    def verify_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        """
        Verifies a new hypothesis against the knowledge graph for direct contradictions.

        :param hypothesis: A dictionary containing at least 'head', 'tail', and 'relation'.
        :return: True if no direct contradiction is found, False otherwise.
        """
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        relation = hypothesis.get('relation')

        if not all([head, tail, relation]):
            self.logger.warning(f"Hypothesis is missing required keys (head, tail, relation): {hypothesis}")
            return False # Invalid hypothesis is treated as a failed verification

        # --- 1. Check for Direct Inverse Relationship ---
        # Example: If hypothesis is "A causes B", check if "B causes A" exists.
        # This is a simple but powerful contradiction check.

        # Define pairs of inverse relations
        inverse_relations = {
            "causes": "caused_by",
            "caused_by": "causes",
            "enables": "enabled_by",
            "enabled_by": "enables",
            "parent_of": "child_of",
            "child_of": "parent_of",
        }

        inverse_relation = inverse_relations.get(relation)
        if inverse_relation:
            # Check if an edge exists from tail to head with the inverse relation
            if self.kg_manager.edge_exists(source=tail, target=head, relation=inverse_relation):
                self.logger.info(f"Contradiction Found: Hypothesis '{head} {relation} {tail}' is contradicted by existing knowledge '{tail} {inverse_relation} {head}'.")
                return False

        # --- 2. Check for Exclusive Relationships (Future Implementation) ---
        # Example: If hypothesis is "A is_a B", check if A is already "is_a C" where B and C are mutually exclusive.
        # This requires a more complex understanding of ontology, saved for a future version.

        self.logger.info(f"Hypothesis '{head} {relation} {tail}' passed verification (no direct contradictions found).")
        return True

if __name__ == '__main__':
    # Example Usage and testing
    from unittest.mock import MagicMock

    # Mock KGManager for testing
    mock_kg_manager = MagicMock(spec=KGManager)

    # Setup test cases in the mock KG
    # Case 1: A direct contradiction
    mock_kg_manager.edge_exists.side_effect = lambda source, target, relation: (
        source == 'sun' and target == 'plant' and relation == 'causes'
    )

    verifier = SelfVerifier(kg_manager=mock_kg_manager)

    # Test Case 1: Should fail verification
    hypo_contradiction = {'head': 'plant', 'tail': 'sun', 'relation': 'caused_by'}
    print(f"Testing contradiction: {hypo_contradiction}")
    result = verifier.verify_hypothesis(hypo_contradiction)
    print(f"Verification Result: {'Passed' if result else 'Failed'} (Expected: Failed)")
    assert not result

    # Reset mock for Case 2
    mock_kg_manager.edge_exists.side_effect = lambda source, target, relation: False

    # Test Case 2: No contradiction
    hypo_no_contradiction = {'head': 'sunlight', 'tail': 'photosynthesis', 'relation': 'enables'}
    print(f"\nTesting no contradiction: {hypo_no_contradiction}")
    result = verifier.verify_hypothesis(hypo_no_contradiction)
    print(f"Verification Result: {'Passed' if result else 'Failed'} (Expected: Passed)")
    assert result

    # Test Case 3: Invalid hypothesis
    hypo_invalid = {'head': 'A', 'relation': 'relates_to'} # Missing tail
    print(f"\nTesting invalid hypothesis: {hypo_invalid}")
    result = verifier.verify_hypothesis(hypo_invalid)
    print(f"Verification Result: {'Passed' if result else 'Failed'} (Expected: Failed)")
    assert not result
