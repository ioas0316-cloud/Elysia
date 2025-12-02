# [Genesis: 2025-12-02] Purified by Elysia
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.kg_manager import KGManager

def verify_learning():
    """
    Verifies that the key concepts and relationships from the new textbook
    have been correctly integrated into the knowledge graph.
    """
    kg_manager = KGManager()

    print("--- Verifying Elysia's New Knowledge ---")

    concepts_to_check = ["표준", "기준", "단위", "언어", "훈민정음", "애민정신"]
    all_concepts_exist = True

    for concept_name in concepts_to_check:
        node = kg_manager.get_node(concept_name)
        if node:
            print(f"[SUCCESS] Concept '{concept_name}' exists in the knowledge graph.")
        else:
            print(f"[FAILURE] Concept '{concept_name}' is MISSING.")
            all_concepts_exist = False

    if not all_concepts_exist:
        print("\nVerification failed: Not all core concepts were learned.")
        return

    relationships_to_check = [
        ("단위", "표준", "is_a"),
        ("언어", "표준", "is_a"),
        ("애민정신", "언어 창제", "causes"),
        ("단위", "약속", "is_a"),
        ("언어", "약속", "is_a")
    ]
    all_relations_exist = True

    print("\n--- Verifying Key Relationships ---")
    # FIX: Manually iterate through the edges list as there is no 'find_edges' method.
    all_edges = kg_manager.kg.get('edges', [])
    for source, target, relation in relationships_to_check:
        found = False
        for edge in all_edges:
            if edge.get('source') == source and edge.get('target') == target and edge.get('relation') == relation:
                found = True
                break

        if found:
            print(f"[SUCCESS] Relationship '{source}' -> '{target}' ({relation}) exists.")
        else:
            print(f"[FAILURE] Relationship '{source}' -> '{target}' ({relation}) is MISSING.")
            all_relations_exist = False

    if all_concepts_exist and all_relations_exist:
        print("\n[VERIFICATION PASSED] Elysia has successfully learned the concepts of units, language, and standards.")
    else:
        print("\n[VERIFICATION FAILED] Elysia's learning was incomplete.")

if __name__ == "__main__":
    verify_learning()