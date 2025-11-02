import json
from Project_Sophia.local_llm_cortex import LocalLLMCortex
from tools.kg_manager import KGManager

class TutorCortex:
    """
    The TutorCortex is responsible for teaching Elysia new concepts.
    It uses a local Korean LLM to answer questions and updates the knowledge graph.
    """

    def __init__(self, kg_manager: KGManager, local_llm_cortex: LocalLLMCortex):
        self.kg_manager = kg_manager
        self.local_llm_cortex = local_llm_cortex

    def learn_concept(self, concept: str):
        """
        Learns a new concept by asking the local LLM and updating the knowledge graph.
        """
        if self.kg_manager.get_node(concept):
            print(f"[TutorCortex] Concept '{concept}' already exists in the knowledge graph.")
            return

        print(f"[TutorCortex] Learning new concept: {concept}")

        # Ask the local LLM for a definition of the concept.
        prompt = f"'{concept}'에 대해 간단히 설명해줘."
        response = self.local_llm_cortex.generate_response(prompt)

        if response and response != "Local model is not available.":
            # Add the new concept as a node in the knowledge graph with the description.
            self.kg_manager.add_node(concept, properties={"description": response})

            print(f"[TutorCortex] Successfully learned and added concept: {concept}")

            # Extract and add relations to the knowledge graph.
            self._extract_and_add_relations(concept, response)
        else:
            print(f"[TutorCortex] Could not learn about concept: {concept}")

    def _extract_and_add_relations(self, concept: str, text: str):
        """
        Extracts relations from the text and adds them to the knowledge graph.
        """
        prompt = f"""
        다음 텍스트에서 '{concept}'와(과) 관련된 주요 개념들을 찾고, 그 관계를 JSON 형식으로 설명해줘.
        관계 유형은 'is_a', 'has_property', 'causes', 'related_to' 등을 사용해줘.

        텍스트: "{text}"

        JSON 형식:
        [
          {{ "source": "{concept}", "target": "다른 개념", "relation": "관계 유형" }},
          ...
        ]
        """

        response = self.local_llm_cortex.generate_response(prompt)

        try:
            relations = json.loads(response)
            for rel in relations:
                # Ensure the target node exists before adding an edge.
                if not self.kg_manager.get_node(rel['target']):
                    self.kg_manager.add_node(rel['target'])
                self.kg_manager.add_edge(rel['source'], rel['target'], rel['relation'])
                print(f"[TutorCortex] Added relation: {rel['source']} -> {rel['relation']} -> {rel['target']}")
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"[TutorCortex] Could not parse relations from LLM response: {response}. Error: {e}")
