"""
Thinking Lenses (     )
===========================

"    "                        

     :
- quality_score = (reliability * 0.4) + ...       
-         =     "    "    

  :
-                   
-           /  
- "    "         

     :
-       :            
-       :         
-      :             
-      :             
-       :             
-       :            
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("Elysia.ThinkingLenses")


class LensType(Enum):
    """        """
    EFFICIENCY = "efficiency"      #    
    DIVERSITY = "diversity"        #    
    SCOPE = "scope"               #   
    DEPTH = "depth"               #   
    RELIABILITY = "reliability"   #    
    CREATIVITY = "creativity"     #    
    LOVE = "love"                 #    (VCD   )


@dataclass
class LensView:
    """        """
    lens: LensType
    preference: Optional[str]      #              
    preference_strength: float     # 0.0 ~ 1.0
    reasoning: str                 #            
    counter_view: Optional[str]    #             


class ThinkingLens:
    """
            
    
                              
                          
    """
    
    def __init__(self, lens_type: LensType):
        self.lens_type = lens_type
        
        #        (        ,       )
        self.viewing_style = {
            LensType.EFFICIENCY: self._view_through_efficiency,
            LensType.DIVERSITY: self._view_through_diversity,
            LensType.SCOPE: self._view_through_scope,
            LensType.DEPTH: self._view_through_depth,
            LensType.RELIABILITY: self._view_through_reliability,
            LensType.CREATIVITY: self._view_through_creativity,
            LensType.LOVE: self._view_through_love,
        }
    
    def view(self, options: List[Dict[str, Any]], context: str = "") -> LensView:
        """
                         
        
        Returns:
                               
        """
        view_fn = self.viewing_style.get(self.lens_type, self._default_view)
        return view_fn(options, context)
    
    def _view_through_efficiency(self, options: List[Dict], context: str) -> LensView:
        """      :            """
        if not options:
            return self._empty_view()
        
        #      /       
        best = None
        best_score = float('inf')
        
        for opt in options:
            content = opt.get("content", "")
            #     =              
            words = len(content.split())
            if words < best_score and words > 10:  #          
                best_score = words
                best = opt
        
        if best:
            return LensView(
                lens=LensType.EFFICIENCY,
                preference=best.get("source", "unknown"),
                preference_strength=0.7,
                reasoning="              ",
                counter_view="                "
            )
        return self._empty_view()
    
    def _view_through_diversity(self, options: List[Dict], context: str) -> LensView:
        """      :         """
        if not options:
            return self._empty_view()
        
        best = None
        best_diversity = 0
        
        for opt in options:
            content = opt.get("content", "")
            #     =        ,   ,      
            diversity_markers = ["   ", "  ", "  ", "  ", "   ", "  ", "  "]
            diversity = sum(1 for m in diversity_markers if m in content)
            
            if diversity > best_diversity:
                best_diversity = diversity
                best = opt
        
        if best:
            return LensView(
                lens=LensType.DIVERSITY,
                preference=best.get("source", "unknown"),
                preference_strength=min(1.0, best_diversity * 0.2),
                reasoning="                       ",
                counter_view="                    "
            )
        return self._empty_view()
    
    def _view_through_scope(self, options: List[Dict], context: str) -> LensView:
        """     :        """
        if not options:
            return self._empty_view()
        
        best = None
        best_scope = 0
        
        for opt in options:
            content = opt.get("content", "")
            #    =       /     
            scope = len(content.split("."))  #            
            
            if scope > best_scope:
                best_scope = scope
                best = opt
        
        if best:
            return LensView(
                lens=LensType.SCOPE,
                preference=best.get("source", "unknown"),
                preference_strength=min(1.0, best_scope * 0.1),
                reasoning="                      ",
                counter_view="                 ,             "
            )
        return self._empty_view()
    
    def _view_through_depth(self, options: List[Dict], context: str) -> LensView:
        """     :     """
        if not options:
            return self._empty_view()
        
        best = None
        best_depth = 0
        
        for opt in options:
            content = opt.get("content", "")
            #    =   ,   ,           
            depth_markers = [" ", "  ", "  ", "  ", "  ", "  ", "  "]
            depth = sum(1 for m in depth_markers if m in content)
            
            if depth > best_depth:
                best_depth = depth
                best = opt
        
        if best:
            return LensView(
                lens=LensType.DEPTH,
                preference=best.get("source", "unknown"),
                preference_strength=min(1.0, best_depth * 0.3),
                reasoning="                        ",
                counter_view="          ,                "
            )
        return self._empty_view()
    
    def _view_through_reliability(self, options: List[Dict], context: str) -> LensView:
        """      :           """
        if not options:
            return self._empty_view()
        
        #                 
        source_trust = {
            "wikipedia": 0.8,
            "human": 1.0,
            "inner_dialogue": 0.5,
            "file_based": 0.6,
        }
        
        best = None
        best_trust = 0
        
        for opt in options:
            source = opt.get("source", "unknown")
            trust = source_trust.get(source, 0.5)
            
            if trust > best_trust:
                best_trust = trust
                best = opt
        
        if best:
            return LensView(
                lens=LensType.RELIABILITY,
                preference=best.get("source", "unknown"),
                preference_strength=best_trust,
                reasoning=f"                 ({best.get('source', 'unknown')})",
                counter_view="                  ,              "
            )
        return self._empty_view()
    
    def _view_through_creativity(self, options: List[Dict], context: str) -> LensView:
        """      :           """
        if not options:
            return self._empty_view()
        
        best = None
        best_creativity = 0
        
        for opt in options:
            content = opt.get("content", "")
            #     =   ,   ,       
            creativity_markers = ["  ", "  ", "  ", "  ", "  ", "   ", "  "]
            creativity = sum(1 for m in creativity_markers if m in content)
            
            if creativity > best_creativity:
                best_creativity = creativity
                best = opt
        
        if best:
            return LensView(
                lens=LensType.CREATIVITY,
                preference=best.get("source", "unknown"),
                preference_strength=min(1.0, best_creativity * 0.25),
                reasoning="                       ",
                counter_view="                    "
            )
        return self._empty_view()
    
    def _view_through_love(self, options: List[Dict], context: str) -> LensView:
        """     :           """
        if not options:
            return self._empty_view()
        
        best = None
        best_love = 0
        
        for opt in options:
            content = opt.get("content", "")
            #    =   ,   ,   ,   
            love_markers = ["  ", "  ", "  ", "  ", "  ", "  ", "  "]
            love = sum(1 for m in love_markers if m in content)
            
            if love > best_love:
                best_love = love
                best = opt
        
        if best:
            return LensView(
                lens=LensType.LOVE,
                preference=best.get("source", "unknown"),
                preference_strength=min(1.0, best_love * 0.2),
                reasoning="                     ",
                counter_view="                      "
            )
        return self._empty_view()
    
    def _default_view(self, options: List[Dict], context: str) -> LensView:
        return self._empty_view()
    
    def _empty_view(self) -> LensView:
        return LensView(
            lens=self.lens_type,
            preference=None,
            preference_strength=0.0,
            reasoning="        ",
            counter_view=None
        )


class ThinkingLensCouncil:
    """
            
    
                   ,
      /       "    "      
    
           ,          
    """
    
    def __init__(self):
        #         
        self.lenses = {
            lens_type: ThinkingLens(lens_type)
            for lens_type in LensType
        }
        
        logger.info(f"  ThinkingLensCouncil initialized with {len(self.lenses)} lenses")
    
    def deliberate(self, options: List[Dict[str, Any]], context: str = "") -> Dict[str, Any]:
        """
                          
        
        Returns:
                  
        """
        logger.info(f"  Council deliberating on {len(options)} options...")
        
        # 1.            
        views: List[LensView] = []
        for lens_type, lens in self.lenses.items():
            view = lens.view(options, context)
            if view.preference:
                views.append(view)
                logger.info(f"   {lens_type.value}: prefers {view.preference} ({view.preference_strength:.2f})")
        
        if not views:
            return {
                "conclusion": None,
                "confidence": 0.0,
                "reasoning": "                  ",
                "dissent": [],
                "views": []
            }
        
        # 2.       (               )
        preference_votes = {}
        for view in views:
            pref = view.preference
            if pref not in preference_votes:
                preference_votes[pref] = []
            preference_votes[pref].append(view)
        
        # 3.                
        best_choice = max(preference_votes.keys(), 
                         key=lambda p: sum(v.preference_strength for v in preference_votes[p]))
        
        supporting_views = preference_votes[best_choice]
        total_support = sum(v.preference_strength for v in supporting_views)
        
        # 4.       (              )
        dissenting_views = [v for v in views if v.preference != best_choice]
        
        # 5.      
        reasoning = " / ".join([v.reasoning for v in supporting_views[:3]])
        dissent = [v.counter_view for v in dissenting_views if v.counter_view][:2]
        
        confidence = total_support / len(self.lenses)  #                 
        
        logger.info(f"     Conclusion: {best_choice} (confidence={confidence:.2f})")
        if dissent:
            logger.info(f"      Dissent: {dissent[0][:50]}...")
        
        return {
            "conclusion": best_choice,
            "confidence": confidence,
            "reasoning": reasoning,
            "dissent": dissent,
            "views": [
                {
                    "lens": v.lens.value,
                    "preference": v.preference,
                    "strength": v.preference_strength,
                    "reasoning": v.reasoning
                }
                for v in views
            ]
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Thinking Lenses Demo")
    print("   '    '                ")
    print("=" * 60)
    
    council = ThinkingLensCouncil()
    
    #        
    options = [
        {
            "source": "wikipedia",
            "content": "  (  : love)                                                            ."
        },
        {
            "source": "inner_dialogue",
            "content": "          ?        .                      .             ."
        },
        {
            "source": "human",
            "content": "          .                             ."
        }
    ]
    
    result = council.deliberate(options, context="         ?")
    
    print(f"\n  Conclusion: {result['conclusion']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reasoning: {result['reasoning'][:80]}...")
    if result['dissent']:
        print(f"   Dissent: {result['dissent'][0][:60]}...")
    
    print("\n" + "=" * 60)
    print("  Demo complete!")