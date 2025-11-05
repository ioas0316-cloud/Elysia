from tools.kg_manager import KGManager

class KnowledgeEnhancer:
    def __init__(self, kg_manager: KGManager):
        self.kg_manager = kg_manager

    def process_learning_points(self, learning_points: list, image_path: str):
        """
        Processes learning points and associates them with a visual experience.

        Args:
            learning_points: A list of knowledge dictionaries (concepts and relations).
            image_path: The path to the image associated with this learning experience.
        """
        concepts = [p for p in learning_points if p.get('type') == 'concept']
        relations = [p for p in learning_points if p.get('type') == 'relation']

        # Ensure all concept nodes exist before creating relations
        for concept in concepts:
            node_id = concept.get('label')
            if not node_id:
                continue

            if not self.kg_manager.get_node(node_id):
                self.kg_manager.add_node(
                    node_id,
                    description=f"Concept learned from visual experience: {image_path}",
                    category="learned_concept",
                    experience_visual=[image_path]
                )
            else:
                # Append the new experience to existing ones
                existing_node = self.kg_manager.get_node(node_id)
                experiences = existing_node.get('experience_visual', [])
                if not isinstance(experiences, list):
                    experiences = [experiences]
                if image_path not in experiences:
                    experiences.append(image_path)
                self.kg_manager.update_node_properties(
                    node_id,
                    properties={'experience_visual': experiences}
                )

        # Create relations, now that nodes are guaranteed to exist
        for relation in relations:
            source = relation.get('source')
            target = relation.get('target')
            label = relation.get('label')
            if source and target and label:
                self.kg_manager.add_edge(
                    source,
                    target,
                    label,
                    experience_visual=image_path
                )

        print(f"Knowledge graph updated with experience from '{image_path}'.")
        self.kg_manager.save()
