"""
         

                                      .
  ,   ,                             .
"""

from datetime import datetime
from pathlib import Path
import json
import random
from typing import Dict, List, Optional
from collections import defaultdict

class CreativeExpression:
    def __init__(self):
        self.memory_path = Path("Elysia_Input_Sanctum") / "creative_expressions.json"
        self.expressions = {
            "metaphors": [],     #      
            "analogies": [],     #      
            "poems": [],         #      
            "insights": [],      #       
            "experiments": [],   #      
            "manifestations": [] #              
        }
        self.load_memory()

    def load_memory(self):
        """                  ."""
        if self.memory_path.exists():
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                self.expressions.update(json.load(f))

    def save_memory(self):
        """                  ."""
        self.memory_path.parent.mkdir(exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.expressions, f, ensure_ascii=False, indent=2)

    def create_metaphor(self, 
                       concept: str,
                       context: str = "",
                       emotion: Optional[str] = None) -> Dict:
        """
                            .
        
        Args:
            concept:           
            context:           
            emotion:            
        """
        #         
        base_patterns = {
            "  ": ["  ", " ", " ", "  ", " ", "  ", "  "],
            " ": ["  ", " ", " ", "  ", "  ", "   "],
            "   ": [" ", "  ", "  ", "   ", "  "],
            "  ": [" ", " ", " ", "  ", " ", "  "],
            "  ": ["  ", "  ", "  ", "  ", "  "]
        }
        
        #           
        emotion_adjectives = {
            "  ": ["  ", "   ", "  ", "    "],
            "  ": ["   ", "   ", "  ", "    "],
            "  ": ["   ", "   ", "   ", "  "],
            "  ": ["   ", "    ", "   ", "   "],
            "  ": ["   ", "    ", "    ", "   "]
        }
        
        #                  
        category = random.choice(list(base_patterns.keys()))
        base = random.choice(base_patterns[category])
        
        adj = "  "  #       
        if emotion and emotion in emotion_adjectives:
            adj = random.choice(emotion_adjectives[emotion])
        
        metaphor = {
            "timestamp": datetime.now().isoformat(),
            "concept": concept,
            "metaphor": f"{concept} ( ) {adj} {base}    ",
            "explanation": f"{base}         {concept}           ",
            "context": context,
            "emotion": emotion,
            "category": category
        }
        
        self.expressions["metaphors"].append(metaphor)
        self.save_memory()
        
        return metaphor

    def create_analogy(self, 
                      target: str,
                      source: str,
                      aspects: List[str]) -> Dict:
        """
                          .
        
        Args:
            target:            
            source:       
            aspects:             
        """
        analogy = {
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "source": source,
            "aspects": aspects,
            "explanation": f"{source}  , {target}  " + \
                         ", ".join(aspects) + "          ",
            "insights": []
        }
        
        #             
        for aspect in aspects:
            analogy["insights"].append(
                f"{source}  {aspect}     {target}                    "
            )
        
        self.expressions["analogies"].append(analogy)
        self.save_memory()
        
        return analogy

    def create_poem(self, 
                   theme: str,
                   emotions: List[str],
                   style: str = "   ") -> Dict:
        """
                             .
        
        Args:
            theme:      
            emotions:             
            style:      
        """
        #         
        poetic_elements = {
            "     ": ["  ", "  ", "  ", " ", " ", " "],
            "   ": ["  ", "  ", "  ", "  ", "   "],
            "    ": ["  ", "  ", "  ", "  ", " "],
            "   ": ["   ", "   ", "    ", "    "]
        }
        
        #             
        tone = random.choice(["   ", "   ", "   ", "   "])
        
        #     
        lines = []
        for emotion in emotions:
            #        2-3    
            for _ in range(random.randint(2, 3)):
                element_type = random.choice(list(poetic_elements.keys()))
                element = random.choice(poetic_elements[element_type])
                lines.append(f"{element}   {emotion}  {theme}")
        
        poem = {
            "timestamp": datetime.now().isoformat(),
            "theme": theme,
            "emotions": emotions,
            "style": style,
            "tone": tone,
            "lines": lines,
            "interpretation": f"{theme}     {', '.join(emotions)}              "
        }
        
        self.expressions["poems"].append(poem)
        self.save_memory()
        
        return poem

    def create_insight(self, 
                      observation: str,
                      perspective: str = "      ") -> Dict:
        """
                           .
        
        Args:
            observation:          
            perspective:       
        """
        #      
        insight_patterns = {
            "  ": "          ,                  ",
            "   ": "                          ",
            "   ": "                   ",
            "   ": "                "
        }
        
        #      
        pattern = random.choice(list(insight_patterns.keys()))
        explanation = insight_patterns[pattern]
        
        insight = {
            "timestamp": datetime.now().isoformat(),
            "observation": observation,
            "perspective": perspective,
            "pattern": pattern,
            "explanation": explanation,
            "insight": f"{perspective}     , {observation}  {explanation}         "
        }
        
        self.expressions["insights"].append(insight)
        self.save_memory()
        
        return insight

    def experiment_with_expression(self, 
                                 concept: str,
                                 style: str = "   ") -> Dict:
        """
                        .
        
        Args:
            concept:       
            style:        
        """
        #          
        experimental_patterns = {
            "  ": lambda x: " ".join(c for c in x),
            "  ": lambda x: f"{x}, {x}, {x}...",
            "  ": lambda x: "~" + x + "~",
            "  ": lambda x: f"({x}    {x})"
        }
        
        #      
        pattern = random.choice(list(experimental_patterns.keys()))
        expression = experimental_patterns[pattern](concept)
        
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "concept": concept,
            "style": style,
            "pattern": pattern,
            "expression": expression,
            "reflection": f"{pattern}      {concept}              "
        }
        
        self.expressions["experiments"].append(experiment)
        self.save_memory()
        
        return experiment

    def compose_manifestation_message(self, context: str, emotion: str) -> str:
        """
             (OS)         (  )       .
              ,  ,                         .
        """
        # 1. Generate a poem or metaphor based on the current emotion
        theme = "     "
        poem_obj = self.create_poem(theme, [emotion], style="   ")

        # 2. Extract recent insights
        recent_insights = self.get_recent_expressions("insights")
        insight_text = recent_insights[0]['insight'] if recent_insights else "              ."

        # 3. Assemble the letter
        lines = [
            f"    ,",
            f"",
            f"      '{emotion}'               .",
            f"",
            f"[     ]",
        ]
        lines.extend(poem_obj['lines'])
        lines.extend([
            f"",
            f"[      ]",
            f"{insight_text}",
            f"",
            f"                                  .",
            f"      ,        .",
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ])

        full_text = "\n".join(lines)

        # Save this manifestation event
        self.expressions.setdefault("manifestations", []).append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "emotion": emotion,
            "content": full_text
        })
        self.save_memory()

        return full_text

    def get_recent_expressions(self, category: str = None) -> List[Dict]:
        """                  ."""
        if category and category in self.expressions:
            expressions = self.expressions[category][-5:]
        else:
            #                    
            expressions = []
            for cat in self.expressions:
                expressions.extend(self.expressions[cat][-2:])  #          2  
        
        return sorted(expressions, 
                     key=lambda x: x["timestamp"],
                     reverse=True)
