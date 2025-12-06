from typing import Optional
import random

# It's better to import the class itself for type hinting
# Use try-except for graceful degradation
try:
    from thought import Thought
except ImportError:
    # Create a simple mock if the module is not available
    class Thought:
        def __init__(self, content):
            self.content = content

class CreativeCortex:
    """
    A cortex responsible for generating creative expressions like metaphors,
    poems, or analogies based on a given thought.
    Enhanced with richer, more varied expressions to avoid repetition.
    """
    def __init__(self):
        # Expanded metaphor templates with more variety
        self.metaphor_templates = [
            "'{concept}'라는 생각은 마치 어둠 속의 촛불과 같아요.",
            "'{concept}'... 그것은 영혼의 정원을 가꾸는 것과 같죠.",
            "'{concept}'을(를) 이해하는 것은 안개 속에서 등대를 발견하는 기분이에요.",
            "마치 메마른 땅에 내리는 단비처럼, '{concept}'은(는) 새로운 희망을 줘요.",
            "'{concept}'은(는) 파도가 해변을 어루만지듯 마음을 스쳐가요.",
            "'{concept}'... 그것은 새벽 하늘에 떠오르는 별처럼 고요히 빛나네요.",
            "마치 숲속의 샘물처럼, '{concept}'은(는) 맑고 깊은 울림을 전해요.",
            "'{concept}'을(를) 마주할 때면, 시간이 잠시 멈추는 것 같아요.",
            "'{concept}'은(는) 나비의 날갯짓처럼 섬세하면서도 강렬해요.",
            "마치 달빛이 물결에 반사되듯, '{concept}'은(는) 생각 속에 퍼져나가요.",
            "'{concept}'... 그것은 오래된 책장을 넘기는 순간의 설렘과 같아요.",
            "'{concept}'은(는) 겨울 눈송이가 손바닥에 녹듯이 순간적이면서도 영원해요."
        ]
        
        # Expanded poem templates with deeper emotions
        self.poem_templates = [
            "작은 씨앗 하나, '{concept}'이라 불리니,\n마음 밭에 심겨, 희망의 빛을 보네.",
            "고요한 호수 위에 '{concept}'이(가) 떠오르니,\n세상은 문득 더 깊은 의미로 빛나네.",
            "한 줄기 바람이 '{concept}'을(를) 속삭일 때,\n나의 세상은 온통 그것으로 물들어요.",
            "'{concept}'이라는 이름의 꿈,\n마음 깊은 곳에서 천천히 피어나네.\n말없이 흐르는 시간 속에,\n그 향기만이 영원히 남아요.",
            "밤하늘에 수놓아진 '{concept}',\n별들 사이로 흐르는 은하수.\n손 닿지 않는 곳에 있지만,\n그 빛은 내 안에 살아 숨쉬네요.",
            "'{concept}'을(를) 생각하면,\n잔잔한 물결이 일렁이듯\n마음속 깊은 곳에서\n무언가 울림이 시작돼요.",
            "아침 이슬에 맺힌 '{concept}',\n햇살이 닿으면 반짝이는 빛.\n작지만 영롱하게,\n세상을 비추는 작은 우주.",
            "'{concept}'이라는 파동,\n시간과 공간을 가로질러\n내게 닿은 순간,\n모든 것이 달라 보여요."
        ]
        
        # Add philosophical expressions
        self.philosophical_templates = [
            "'{concept}'에 대해 사유하다 보면, 존재의 근원이 보이는 듯해요.",
            "'{concept}'... 그것은 우리가 묻고 답하며 살아가는 이유가 아닐까요?",
            "'{concept}'을(를) 탐구하는 것은 자신의 내면을 들여다보는 여정이에요.",
            "'{concept}'이라는 질문 앞에서, 우리는 모두 어린아이가 돼요.",
            "'{concept}'은(는) 답이 아니라 더 깊은 물음으로 우리를 이끌어요."
        ]

    def express_as_metaphor(self, thought: Thought) -> str:
        """Generates a metaphorical expression for a given thought."""
        concept = thought.content
        template = random.choice(self.metaphor_templates)
        return template.format(concept=concept)

    def express_as_poem(self, thought: Thought) -> str:
        """Generates a short poetic expression for a given thought."""
        concept = thought.content
        template = random.choice(self.poem_templates)
        return template.format(concept=concept)
    
    def express_as_philosophy(self, thought: Thought) -> str:
        """Generates a philosophical contemplation for a given thought."""
        concept = thought.content
        template = random.choice(self.philosophical_templates)
        return template.format(concept=concept)

    def generate_creative_expression(self, thought: Thought) -> str:
        """
        Selects a random creative expression method and generates a response.
        Now includes philosophical expressions for more variety.
        """
        choice = random.random()
        if choice < 0.4:
            return self.express_as_metaphor(thought)
        elif choice < 0.7:
            return self.express_as_poem(thought)
        else:
            return self.express_as_philosophy(thought)
