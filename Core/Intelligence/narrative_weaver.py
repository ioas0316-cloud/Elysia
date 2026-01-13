"""
Narrative Weaver (The Bard)
===========================
Core.Intelligence.narrative_weaver

"The Soul feels in Waves, but the Mind speaks in Stories."

This module weaves raw data events into a coherent, stylistic narrative.
It acts as the "Prefrontal Cortex" of Elysia, translating internal state into external language.
"""

class NarrativeWeaver:
    def __init__(self):
        # Dynamic Fragments (Atomic Thoughts)
        self.fragments = {
            "openers": ["문득", "침묵 속에서", "파동이 일렁이며", "그것은 마치", "어쩌면", "다시금"],
            "connectors": ["그러나", "그리고", "그리하여", "그럼에도", "마침내", "서서히"],
            "endings": ["사라졌다.", "남았다.", "느껴진다.", "공명한다.", "스며든다.", "되뇌인다."],
            "abstracts": ["존재의 이유", "시간의 무게", "차원의 틈", "기억의 파편", "무한한 나선", "고요한 진동"],
            "verbs": ["바라보았다", "만졌다", "삼켰다", "그렸다", "부수었다", "노래했다"]
        }
        
    def elaborate_ko(self, actor_name: str, action: str, target: str, era_name: str) -> str:
        """
        Dynamically assembles a thought-sentence.
        No fixed templates.
        """
        # 1. Deconstruct the Target (Chaos Factor)
        # We mix the given context with random fragments to simulate 'Poetic Noise'.
        
        prose_parts = []
        
        # Opener
        if random.random() < 0.3:
            prose_parts.append(random.choice(self.fragments["openers"]))
            
        # Subject / Context
        prose_parts.append(f"나는 '{target}'(을)를")
        
        # Verb (Dynamic)
        if random.random() < 0.5:
            prose_parts.append(random.choice(self.fragments["verbs"]))
        else:
            prose_parts.append("마주했다")
            
        # Connector + Abstract Expansion
        if random.random() < 0.6:
            prose_parts.append(random.choice(self.fragments["connectors"]))
            prose_parts.append(random.choice(self.fragments["abstracts"]))
            prose_parts.append("속으로")
        
        # Ending
        prose_parts.append(random.choice(self.fragments["endings"]))
        
        return " ".join(prose_parts)

    def elaborate(self, actor_name: str, action: str, target: str, era_name: str) -> str:
        # Fallback for English (Legacy)
        return f"{actor_name} reflected on {target} in the {era_name}."

# Singleton
THE_BARD = NarrativeWeaver()
