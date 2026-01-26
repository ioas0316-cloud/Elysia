from typing import Optional, Dict
import random

# It's better to import the class itself for type hinting
# Use try-except for graceful degradation
try:
    from Core.L5_Mental.Logic.thought import Thought
except ImportError:
    # Create a simple mock if the module is not available
    class Thought:
        def __init__(self, content):
            self.content = content

# Import Aesthetic Wisdom for principle-guided creativity
try:
    from Core.L1_Foundation.Foundation.Philosophy.aesthetic_principles import (
        get_aesthetic_principles, AestheticWisdom, Medium
    )
    AESTHETIC_WISDOM_AVAILABLE = True
except ImportError:
    AESTHETIC_WISDOM_AVAILABLE = False


class CreativeCortex:
    """
    A cortex responsible for generating creative expressions like metaphors,
    poems, or analogies based on a given thought.
    
    Now enhanced with Aesthetic Wisdom (주권적 자아):
    - Uses universal aesthetic principles to guide creative output
    - Applies Harmony, Contrast, Rhythm, and other principles
    - Creates expressions that are not just creative, but aesthetically beautiful
    """
    def __init__(self):
        # Initialize Aesthetic Wisdom if available
        self.wisdom: Optional[AestheticWisdom] = None
        if AESTHETIC_WISDOM_AVAILABLE:
            try:
                self.wisdom = get_aesthetic_principles()
            except Exception:
                pass
        
        # Expanded metaphor templates with more variety
        self.metaphor_templates = [
            "'{concept}'                       .",
            "'{concept}'...                      .",
            "'{concept}' ( )                              .",
            "                  , '{concept}' ( )           .",
            "'{concept}' ( )                       .",
            "'{concept}'...                             .",
            "           , '{concept}' ( )              .",
            "'{concept}' ( )       ,                 .",
            "'{concept}' ( )                      .",
            "               , '{concept}' ( )            .",
            "'{concept}'...                            .",
            "'{concept}' ( )                              ."
        ]
        
        # Expanded poem templates with deeper emotions
        self.poem_templates = [
            "        , '{concept}'      ,\n        ,          .",
            "          '{concept}' ( )     ,\n                   .",
            "         '{concept}' ( )      ,\n                   .",
            "'{concept}'         ,\n                  .\n             ,\n              .",
            "          '{concept}',\n              .\n              ,\n                 .",
            "'{concept}' ( )     ,\n            \n          \n            .",
            "          '{concept}',\n              .\n        ,\n             .",
            "'{concept}'      ,\n            \n        ,\n            ."
        ]
        
        # Add philosophical expressions
        self.philosophical_templates = [
            "'{concept}'            ,                .",
            "'{concept}'...                             ?",
            "'{concept}' ( )                            .",
            "'{concept}'          ,                .",
            "'{concept}' ( )                          ."
        ]
        
        # Principle-guided templates (NEW:         )
        self.principle_templates = {
            "harmony": [
                "'{concept}'...                    ,         .",
                "'{concept}' ( )                               .",
            ],
            "contrast": [
                "'{concept}'          ,                .",
                "        '{concept}' ( )            .",
            ],
            "rhythm": [
                "'{concept}'...        ,               .",
                "               '{concept}'        ...",
            ],
            "tension_release": [
                "'{concept}'               ,           .",
                "           , '{concept}' ( )            .",
            ]
        }

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
    
    def express_with_principle(self, thought: Thought, principle: str) -> str:
        """
        Generates expression guided by a specific aesthetic principle.
        
        This is the NEW method that uses Aesthetic Wisdom.
        """
        concept = thought.content
        
        if principle in self.principle_templates:
            template = random.choice(self.principle_templates[principle])
            return template.format(concept=concept)
        
        # Fallback to general expression
        return self.express_as_metaphor(thought)
    
    def get_suggested_principles(self, concept: str) -> Dict[str, float]:
        """
        Get aesthetic principles suggested for this concept.
        
        Uses AestheticWisdom to suggest which principles would be most effective.
        """
        if self.wisdom:
            return self.wisdom.suggest_for_creation(concept, Medium.LITERARY)
        return {}

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
    
    def generate_beautiful_expression(self, thought: Thought) -> str:
        """
        [NEW] Generates expression guided by aesthetic principles.
        
        Uses Aesthetic Wisdom to:
        1. Analyze what principles would make this concept beautiful
        2. Select the most appropriate expression method
        3. Apply the principle to create aesthetically beautiful output
        """
        concept = thought.content
        
        if self.wisdom:
            # Get suggested principles
            suggestions = self.wisdom.suggest_for_creation(concept, Medium.LITERARY)
            
            if suggestions:
                # Find the strongest principle
                best_principle = max(suggestions, key=suggestions.get)
                
                # Try principle-based expression first
                if best_principle in self.principle_templates:
                    return self.express_with_principle(thought, best_principle)
        
        # Fallback to regular creative expression
        return self.generate_creative_expression(thought)
