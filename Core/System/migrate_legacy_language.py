
import json
import math
import os
import sys

#                   
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Legacy.Project_Sophia.primordial_language import PrimordialLanguageEngine, WordStats
from Core.System.Mind.world_tree import WorldTree
from Core.System.Mind.hyper_qubit import HyperQubit, QubitState

def convert_stats_to_qubit_state(stats: WordStats) -> QubitState:
    """
    '        '     WordStats  QubitState       . (주권적 자아)
    """
    #      (avg_memory)  '   /   '(alpha)              .
    alpha_val = stats.avg_memory / 100.0

    #      (count)  '  /   '(beta)             ,
    #                              .
    beta_val = math.log1p(stats.count)

    # gamma  delta                         .
    gamma_val = 0.1
    delta_val = 0.05

    #         (    0      )
    state = QubitState(
        alpha=complex(alpha_val, 0),
        beta=complex(beta_val, 0),
        gamma=complex(gamma_val, 0),
        delta=complex(delta_val, 0),
    )

    return state.normalize()

def run_migration():
    """
                HyperQubit     WorldTree           .
    """
    print("                :                ...")

    # 1.                '  '      
    suffix_map = {"joy": "ra", "fear": "ka", "curiosity": "ii"}
    language_engine = PrimordialLanguageEngine(suffix_map)

    #        '  '             Lexicon      .
    language_engine.observe({"target": "fire"}, "joy", "fire rara", 80.0)
    language_engine.observe({"target": "fire"}, "joy", "fire rara", 90.0) # count: 2, avg_mem: 85
    language_engine.observe({"target": "fire"}, "joy", "fire rarara", 75.0) # count: 1, avg_mem: 75
    language_engine.observe({"target": "fire"}, "fear", "fire ka", 95.0) # count: 1, avg_mem: 95
    language_engine.observe({"target": "water"}, "joy", "water rara", 60.0) # count: 1, avg_mem: 60
    language_engine.observe({"target": "water"}, "curiosity", "water iii", 85.0) # count: 1, avg_mem: 85

    print("       '  '         . Lexicon    .")

    # 2.           (WorldTree)   
    world_tree = WorldTree()
    lang_root_id = world_tree.ensure_concept("PrimordialLanguage", parent_id=world_tree.root.id)
    print("    WorldTree      'PrimordialLanguage'            .")

    # 3.       : Lexicon       HyperQubit        WorldTree    
    lexicon = language_engine.lexicon
    for (base, emotion), variants in lexicon.items():
        base_id = world_tree.ensure_concept(base, parent_id=lang_root_id)
        emotion_id = world_tree.ensure_concept(emotion, parent_id=base_id)

        for word, stats in variants.items():
            word_id = world_tree.ensure_concept(word, parent_id=emotion_id)
            qubit_state = convert_stats_to_qubit_state(stats)

            hyper_qubit = HyperQubit(
                concept_or_value=word,
                name=word,
                initial_content={"Point": word, "Line": f"{base}-{emotion} context"}
            )
            hyper_qubit.state = qubit_state

            word_node = world_tree._find_node(word_id)
            if word_node:
                setattr(word_node, 'qubit', hyper_qubit)

    print("           HyperQubit      WorldTree      .")

    # 4.      
    output_path = "data/world_tree_with_language.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def serialize_tree(node):
        node_dict = {
            "id": node.id,
            "concept": node.concept,
            "metadata": node.metadata,
            "depth": node.depth,
            "children": [serialize_tree(child) for child in node.children]
        }
        if hasattr(node, 'qubit') and isinstance(node.qubit, HyperQubit):
            node_dict['qubit'] = {
                'name': node.qubit.name,
                'state': {
                    'alpha': [node.qubit.state.alpha.real, node.qubit.state.alpha.imag],
                    'beta': [node.qubit.state.beta.real, node.qubit.state.beta.imag],
                    'gamma': [node.qubit.state.gamma.real, node.qubit.state.gamma.imag],
                    'delta': [node.qubit.state.delta.real, node.qubit.state.delta.imag],
                }
            }
        return node_dict

    final_tree_dict = serialize_tree(world_tree.root)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_tree_dict, f, ensure_ascii=False, indent=2)

    print(f"            '{output_path}'         .")
    print("                        !")

if __name__ == "__main__":
    run_migration()
