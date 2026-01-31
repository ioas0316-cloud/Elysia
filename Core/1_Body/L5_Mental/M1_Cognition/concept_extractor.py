# -*- coding: utf-8 -*-
"""
ConceptExtractor -         
===================================

                      ,   ,       
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

# Optional import
try:
    from Core.1_Body.L5_Mental.Reasoning_Core.Intelligence.korean_mapping import get_korean_name
except ImportError:
    def get_korean_name(name: str) -> str:
        return name  # Fallback: return as-is

logger = logging.getLogger("ConceptExtractor")

@dataclass
class ConceptDefinition:
    """     """
    name: str
    kr_name: str = ""  #       
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    type: str = "general"  # emotion, action, object, abstract...
    context: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'kr_name': self.kr_name,
            'description': self.description,
            'properties': self.properties,
            'type': self.type
        }


class ConceptExtractor:
    """               (   /     )"""
    
    #       (   +    )
    DEFINITION_PATTERNS = [
        #   
        r"(\w+) is (an? )?(.+?)(?:\.|,|;|$)",  # X is Y
        r"(\w+) means (.+?)(?:\.|,|;|$)",      # X means Y
        r"(\w+): (.+?)(?:\.|,|;|$)",           # X: Y
        
        #     -   
        r"(.+?)[    ]\s+(.+?)  ",           # X / / /  Y  
        r"(.+?)[  ]\s+(.+?)   ",            # X /  Y   
        r"(.+?)(?: )? \s+(.+?)  ",           # X  Y  
        r"(.+?)(?: )?  \s+  \s+(.+)",       # X      Y
        r"(.+?)[  ]\s+(.+?)(?: )?  \s+  ", # X  Y     
    ]
    
    #       (   +    )
    PROPERTY_PATTERNS = [
        #   
        r"(\w+) has (.+?)(?:\.|,|;|$)",        # X has Y
        r"(\w+) (?:is|are) (\w+) (?:and|,)",   # X is adj
        
        #    
        r"(.+?)[    ]\s+(.+?)(?:  |  |  )",  # X  Y  
        r"(.+?)[ ]\s+(.+?)[    ]",                # X  Y 
    ]
    
    #           (   +    )
    TYPE_KEYWORDS = {
        'emotion': [
            'feel', 'emotion', 'affection', 'feeling',
            '  ', '  ', '  ', '  ', '  ', '  ', '  ', ' '
        ],
        'action': [
            'do', 'make', 'create', 'move', 'go',
            '  ', '   ', '  ', '  ', '    '
        ],
        'object': [
            'thing', 'item', 'object',
            ' ', '  ', '  '
        ],
        'abstract': [
            'concept', 'idea', 'principle',
            '  ', '  ', '  ', '  '
        ],
        'relation': [
            '  ', '  ', '  ', '  ', '  ', '  '
        ],
        'wisdom': [
            '  ', '  ', '  ', '  '
        ],
    }
    
    def __init__(self):
        self.stopwords = {
            #   
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'as', 'by', 'with', 'from', 'is', 'are',
            #     (  ,   )
            ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
            ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
        }
    
    def extract_concepts(self, text: str) -> List[ConceptDefinition]:
        """           """
        concepts = []
        
        #      
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            #         
            for pattern in self.DEFINITION_PATTERNS:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    name = match.group(1).strip()
                    
                    #       
                    if name.lower() in self.stopwords:
                        continue
                    
                    #      
                    if len(match.groups()) >= 3:
                        description = match.group(3).strip()
                    else:
                        description = match.group(2).strip() if len(match.groups()) >= 2 else ""
                    
                    #         
                    concept_type = self._infer_type(sentence)
                    
                    #      
                    properties = self._extract_properties(sentence)
                    
                    concept = ConceptDefinition(
                        name=name,
                        kr_name=get_korean_name(name),  #       !
                        description=description,
                        properties=properties,
                        type=concept_type,
                        context=sentence
                    )
                    
                    concepts.append(concept)
                    logger.info(f"  Concept: {name} = {description[:50]}...")
        
        #       (     )
        unique_concepts = {}
        for c in concepts:
            if c.name not in unique_concepts:
                unique_concepts[c.name] = c
            else:
                #              
                if len(c.description) > len(unique_concepts[c.name].description):
                    unique_concepts[c.name] = c
        
        return list(unique_concepts.values())
    
    def _infer_type(self, text: str) -> str:
        """             """
        text_lower = text.lower()
        
        for concept_type, keywords in self.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return concept_type
        
        return "general"
    
    def _extract_properties(self, text: str) -> Dict[str, Any]:
        """           """
        properties = {}
        
        #           
        # "X is Y"      Y           
        adjectives = ['positive', 'negative', 'high', 'low', 'intense', 
                     'deep', 'strong', 'weak', 'big', 'small']
        
        for adj in adjectives:
            if adj in text.lower():
                #             
                if adj in ['positive', 'negative']:
                    properties['valence'] = adj
                elif adj in ['high', 'low', 'intense', 'deep', 'strong', 'weak']:
                    properties['intensity'] = adj
                elif adj in ['big', 'small']:
                    properties['size'] = adj
        
        return properties


#    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = ConceptExtractor()
    
    test_text = """
    Love is an intense feeling of deep affection.
    Love creates emotional bonds between people.
    Freedom means the power to act without constraint.
    """
    
    concepts = extractor.extract_concepts(test_text)
    
    print("\n        :")
    for c in concepts:
        print(f"\n  : {c.name}")
        print(f"  : {c.description}")
        print(f"  : {c.properties}")
        print(f"  : {c.type}")
