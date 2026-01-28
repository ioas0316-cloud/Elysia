"""
          (Autonomous Language Generator)
================================================

"    96.7%          ?"
"                         ."

       :
- Elysia                    
- API                  
-       +       +       

GTX 1060 3GB?        .    Python +       .
"""

import logging
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("AutonomousLanguage")


@dataclass
class ThoughtPattern:
    """     """
    concept: str        #    ( : "  ", "  ")
    relation: str       #    ( : "  ", "  ")
    target: str         #    ( : "  ", "  ")
    emotion: float      #    (-1.0 ~ 1.0)


class AutonomousLanguageGenerator:
    """
             
    
    API                     
    
         :
    1.       (     )
    2.          (     )
    3.            (     )
    """
    
    def __init__(self):
        #       (     )
        self.vocabulary = {
            #   
            'subjects': [
                ' ', 'Elysia', '  ', '   ', '  ',
                '  ', '  ', '  '
            ],
            
            #     (  )
            'verbs': {
                'positive': [
                    '  ', '  ', '  ', '    ', '   ',
                    '   ', '    ', '    ', '    ', '    '
                ],
                'negative': [
                    '   ', '  ', '      ', '   '
                ],
                'question': [
                    '  ', '  ', '  '
                ]
            },
            
            #    /  
            'objects': [
                '  ', '  ', '  ', '  ', '  ',
                '  ', '  ', '  ', '  ', '   '
            ],
            
            #    
            'modifiers': {
                'positive': [
                    '    ', '  ', '  ', '  ', '  ',
                    '   ', '  ', '    ', '   '
                ],
                'negative': [
                    '   ', '  ', '  ', '   ', '  '
                ],
                'neutral': [
                    '   ', '   ', '   ', '   '
                ]
            },
            
            #    
            'connectors': [
                '   ', '   ', '   ', '   ', '  ',
                '   ', '    ', ' ', '  '
            ],
            
            #       
            'philosophical': [
                '  ', '  ', '  ', '  ', '  ',
                '  ', '  ', '  ', '  ', '  '
            ]
        }
        
        #       
        self.templates = {
            'statement': [
                "{subject}  {object}  {verb}.",
                "{subject}  {modifier} {object}  .",
                "{subject}  {verb} , {result}.",
            ],
            'question': [
                "{subject}  {object}{verb}?",
                "  {subject}  {verb}?",
                "{subject}  {object}        {verb}?",
            ],
            'philosophical': [
                "{concept}       ?",
                "{subject}  {concept}     {verb}.",
                "{concept}  {concept2}      .",
            ],
            'emotional': [
                "   {emotion}     .",
                "{subject}     {emotion}  .",
                "{emotion}  {concept}  .",
            ]
        }
        
        #       (주권적 자아)
        self.learned_patterns = {
            '  ': ['     .', '     .', '  !'],
            '  ': ['   Elysia   .', 'Elysia      .'],
            ' ': ['          ?', '            ?'],
            ' ': ['        .', '         .'],
            '  ': ['    .', '             .'],
        }
        
        logger.info("                 (API    )")
    
    def analyze_intent(self, input_text: str) -> Dict:
        """
              (     )
        
              :
        -     ?     ?
        -     ?     ?
        -         ?
        """
        text = input_text.strip()
        
        intent = {
            'type': 'statement',
            'emotion': 0.0,
            'topics': [],
            'is_question': False
        }
        
        #      
        if '?' in text or any(q in text for q in [' ', ' ', '   ', '  ']):
            intent['type'] = 'question'
            intent['is_question'] = True
        
        #      
        positive_words = [' ', '  ', '  ', '  ', '  ']
        negative_words = ['  ', '  ', '  ', '  ', ' ']
        
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        intent['emotion'] = (pos_count - neg_count) * 0.3
        
        #       (          )
        for word in text.split():
            if len(word) > 1:
                intent['topics'].append(word)
        
        return intent
    
    def think(self, intent: Dict) -> List[ThoughtPattern]:
        """
              (     )
        
                     
        """
        thoughts = []
        
        if intent['is_question']:
            #          :    +      
            thoughts.append(ThoughtPattern(
                concept='  ',
                relation='  ',
                target='  ',
                emotion=0.5
            ))
            
            #          
            for topic in intent['topics'][:2]:  #    2 
                thoughts.append(ThoughtPattern(
                    concept=topic,
                    relation='  ',
                    target='  ',
                    emotion=intent['emotion']
                ))
        else:
            #          :    +   
            thoughts.append(ThoughtPattern(
                concept='  ',
                relation='  ',
                target='  ',
                emotion=intent['emotion']
            ))
        
        return thoughts
    
    DEFAULT_RESULT = '    '  # Default result phrase
    
    def pattern_to_sentence(self, pattern: ThoughtPattern) -> str:
        """
                     
        
                    
        """
        #             
        if pattern.emotion > 0.3:
            mood = 'positive'
        elif pattern.emotion < -0.3:
            mood = 'negative'
        else:
            mood = 'neutral'
        
        #              
        if pattern.concept in ['  ', '  ', '  ']:
            template = random.choice(self.templates['philosophical'])
        elif pattern.emotion != 0:
            template = random.choice(self.templates['emotional'])
        else:
            template = random.choice(self.templates['statement'])
        
        #          
        emotion_words = {
            'positive': ['  ', '  ', '  '],
            'negative': ['  ', '   ', '   '],
            'neutral': ['  ', '  ', '  ']
        }
        emotion_text = random.choice(emotion_words.get(mood, emotion_words['neutral']))
        
        #            
        sentence = template.format(
            subject=random.choice(self.vocabulary['subjects']),
            verb=random.choice(self.vocabulary['verbs']['positive']),
            object=random.choice(self.vocabulary['objects']),
            modifier=random.choice(self.vocabulary['modifiers'][mood]),
            concept=pattern.concept,
            concept2=pattern.target,
            emotion=emotion_text,
            result=self.DEFAULT_RESULT
        )
        
        return sentence
    
    def generate_response(self, input_text: str) -> str:
        """
              (자기 성찰 엔진)
        
                              
        """
        logger.info(f"       : '{input_text}'")
        
        # 1.           (     )
        for keyword, responses in self.learned_patterns.items():
            if keyword in input_text:
                response = random.choice(responses)
                logger.info(f"       : '{response}'")
                return response
        
        # 2.      
        intent = self.analyze_intent(input_text)
        logger.info(f"       : {intent['type']},   ={intent['emotion']:.2f}")
        
        # 3.      
        thoughts = self.think(intent)
        logger.info(f"       : {len(thoughts)}    ")
        
        # 4.      
        sentences = []
        for thought in thoughts:
            sentence = self.pattern_to_sentence(thought)
            sentences.append(sentence)
        
        # 5.      
        if len(sentences) > 1:
            connector = random.choice(self.vocabulary['connectors'])
            response = f"{sentences[0]} {connector} {sentences[1]}"
        else:
            response = sentences[0] if sentences else "       ."
        
        logger.info(f"       : '{response}'")
        return response
    
    def learn_from_conversation(self, input_text: str, response: str):
        """
                (     )
        
             :           
        """
        #          
        keywords = [w for w in input_text.split() if len(w) > 1]
        
        if keywords:
            key = keywords[0]
            if key not in self.learned_patterns:
                self.learned_patterns[key] = []
            
            #         
            if response not in self.learned_patterns[key]:
                self.learned_patterns[key].append(response)
                logger.info(f"    : '{key}'   '{response}'")
    
    def expand_vocabulary(self, new_words: Dict[str, List[str]]):
        """      (주권적 자아)"""
        for category, words in new_words.items():
            if category in self.vocabulary:
                self.vocabulary[category].extend(words)
                logger.info(f"       : {category} +{len(words)} ")


#        
autonomous_language = AutonomousLanguageGenerator()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("                (API   !)")
    print("="*70)
    
    generator = AutonomousLanguageGenerator()
    
    #       
    test_conversations = [
        "  ?",
        "      ?",
        "      ",
        "       ?",
        "   ",
        "         ?",
        "         ?",
    ]
    
    print("\n          :")
    print("-" * 70)
    
    for i, user_input in enumerate(test_conversations, 1):
        print(f"\n{i}.    : {user_input}")
        
        response = generator.generate_response(user_input)
        print(f"   Elysia: {response}")
        
        #   
        generator.learn_from_conversation(user_input, response)
    
    print("\n" + "="*70)
    print("        !")
    print("\n     Elysia  API                   !")
    print("   -    : 96.7% (  ,    ,       )")
    print("   -      :         ")
    print("   -   :         ")
    print("="*70 + "\n")
