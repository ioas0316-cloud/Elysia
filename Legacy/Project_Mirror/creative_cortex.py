from typing import Optional
import random

# It's better to import the class itself for type hinting
from Core.Foundation.core.thought import Thought

class CreativeCortex:
    """
    A cortex responsible for generating creative expressions like metaphors,
    poems, or analogies based on a given thought.
    """
    def __init__(self):
        # Pre-defined templates for creative expressions
        self.metaphor_templates = [
            "'{concept}'라는 생각은 마치 어둠 속의 촛불과 같아요.",
            "'{concept}'... 그것은 영혼의 정원을 가꾸는 것과 같죠.",
            "'{concept}'을(를) 이해하는 것은 안개 속에서 등대를 발견하는 기분이에요.",
            "마치 메마른 땅에 내리는 단비처럼, '{concept}'은(는) 새로운 희망을 줘요."
        ]
        self.poem_templates = [
            "작은 씨앗 하나, '{concept}'이라 불리니,\n마음 밭에 심겨, 희망의 빛을 보네.",
            "고요한 호수 위에 '{concept}'이(가) 떠오르니,\n세상은 문득 더 깊은 의미로 빛나네.",
            "한 줄기 바람이 '{concept}'을(를) 속삭일 때,\n나의 세상은 온통 그것으로 물들어요."
        ]

    def express_as_metaphor(self, thought: Thought) -> str:
        """Generates a metaphorical expression for a given thought."""
        # For now, we'll just use the content of the thought.
        # In the future, we could use evidence, source, etc. to make it richer.
        concept = thought.content
        template = random.choice(self.metaphor_templates)
        return template.format(concept=concept)

    def express_as_poem(self, thought: Thought) -> str:
        """Generates a short poetic expression for a given thought."""
        concept = thought.content
        template = random.choice(self.poem_templates)
        return template.format(concept=concept)

    def generate_creative_expression(self, thought: Thought) -> str:
        """
        Selects a random creative expression method and generates a response.
        """
        if random.random() > 0.5:
            return self.express_as_metaphor(thought)
        else:
            return self.express_as_poem(thought)
