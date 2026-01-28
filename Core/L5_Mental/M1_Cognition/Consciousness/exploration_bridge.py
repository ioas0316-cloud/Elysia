"""
Exploration Bridge (주권적 자아)
================================

"[     ]"                

     :
- WhyEngine  "[     ]"       
-               (     )

  :
- WhyEngine   FreeWillEngine.Curiosity   
- FreeWillEngine   ExplorationCore      
- ExplorationCore   AutonomousLearner   
-      WhyEngine          

      :
-             
- "     "      ,              ,        
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("Elysia.ExplorationBridge")


class ExplorationDecision(Enum):
    """        """
    EXPLORE = "explore"      #         
    DEFER = "defer"          #        
    ASK_HUMAN = "ask_human"  #         
    SKIP = "skip"            #        


@dataclass
class ExplorationNeed:
    """        """
    question: str
    source: str  #          
    priority: float  # 0.0 ~ 1.0
    domain: str = "general"


@dataclass
class ExplorationResult:
    """     """
    question: str
    answer: Optional[str]
    principle_extracted: Optional[str]
    source: str  # "external", "human", "internal"
    success: bool


@dataclass
class SourceQuality:
    """        """
    source_name: str
    content: Optional[str]
    quality_score: float  # 0.0 ~ 1.0
    reliability: float    #    
    relevance: float      #    
    depth: float          #   


class ExplorationBridge:
    """
           -          
    
      :
    1. WhyEngine   "[     ]"   
    2.          
    3. FreeWillEngine  Curiosity   
    4.       :        ?
    5.       (ExplorationCore, AutonomousLearner)
    6.     WhyEngine         
    """
    
    def __init__(self):
        # ===        ===
        
        # 1. WhyEngine (자기 성찰 엔진)
        self.why_engine = None
        try:
            from Core.L7_Spirit.Philosophy.why_engine import WhyEngine
            self.why_engine = WhyEngine()
            logger.info("  WhyEngine connected")
        except Exception as e:
            logger.warning(f"WhyEngine not available: {e}")
        
        # 2. FreeWillEngine (  /  )
        self.free_will = None
        try:
            from Core.L1_Foundation.M1_Keystone.free_will_engine import FreeWillEngine
            self.free_will = FreeWillEngine()
            logger.info("  FreeWillEngine connected")
        except Exception as e:
            logger.warning(f"FreeWillEngine not available: {e}")
        
        # 3. ExplorationCore (     )
        self.exploration_core = None
        try:
            from Core.L1_Foundation.M1_Keystone.exploration_core import ExplorationCore
            self.exploration_core = ExplorationCore()
            logger.info("  ExplorationCore connected")
        except Exception as e:
            logger.warning(f"ExplorationCore not available: {e}")
        
        # 4. AutonomousLearner (  )
        self.learner = None
        try:
            from Core.L4_Causality.M3_Mirror.Evolution.Learning.Learning.autonomous_learner import AutonomousLearner
            self.learner = AutonomousLearner()
            logger.info("  AutonomousLearner connected")
        except Exception as e:
            logger.warning(f"AutonomousLearner not available: {e}")
        
        # 5. NaverSearchConnector (         )
        self.naver = None
        try:
            from Core.L2_Metabolism.Physiology.Sensory.Network.naver_connector import NaverSearchConnector
            self.naver = NaverSearchConnector()
            if self.naver.available:
                logger.info("  NaverConnector connected")
        except Exception as e:
            logger.warning(f"NaverConnector not available: {e}")
        
        # 6. KoreanEnglishMapper (주권적 자아)
        self.lang_mapper = None
        try:
            from Core.L1_Foundation.M1_Keystone.extreme_hyper_learning import KoreanEnglishMapper
            self.lang_mapper = KoreanEnglishMapper()
            logger.info("  KoreanEnglishMapper connected")
        except Exception as e:
            logger.warning(f"KoreanEnglishMapper not available: {e}")
        
        # 7. PotentialCausalityStore (         )
        self.potential_store = None
        try:
            from Core.L5_Mental.M1_Cognition.Memory_Linguistics.Memory.potential_causality import PotentialCausalityStore
            self.potential_store = PotentialCausalityStore()
            logger.info("  PotentialCausalityStore connected")
        except Exception as e:
            logger.warning(f"PotentialCausalityStore not available: {e}")
        
        #     
        self.exploration_queue: List[ExplorationNeed] = []
        self.exploration_history: List[ExplorationResult] = []
        
        logger.info("  ExplorationBridge initialized")
    
    def detect_exploration_need(self, content: str, subject: str = "unknown") -> Optional[ExplorationNeed]:
        """
        WhyEngine                
        
        "[     ]"       ExplorationNeed   
        """
        if not self.why_engine:
            return None
        
        try:
            analysis = self.why_engine.analyze(
                subject=subject,
                content=content,
                domain="general"
            )
            
            # "[     ]"          
            if "[     ]" in analysis.underlying_principle:
                need = ExplorationNeed(
                    question=content,
                    source="why_engine",
                    priority=1.0 - analysis.confidence,  #                
                    domain="general"
                )
                
                self.exploration_queue.append(need)
                logger.info(f"  Exploration need detected: {content[:50]}...")
                
                return need
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        return None
    
    def stimulate_curiosity(self, need: ExplorationNeed):
        """
        FreeWillEngine  Curiosity      
        
                            
        """
        if not self.free_will:
            return
        
        # Curiosity      
        curiosity_boost = 0.2 + (need.priority * 0.3)  # 0.2 ~ 0.5
        self.free_will.vectors["Curiosity"] = min(
            1.0,
            self.free_will.vectors.get("Curiosity", 0.5) + curiosity_boost
        )
        
        logger.info(f"  Curiosity stimulated: +{curiosity_boost:.2f}   {self.free_will.vectors['Curiosity']:.2f}")
    
    def decide_exploration(self, need: ExplorationNeed) -> ExplorationDecision:
        """
              :        ?
        
        FreeWillEngine           
        """
        if not self.free_will:
            #   :   
            return ExplorationDecision.EXPLORE
        
        curiosity = self.free_will.vectors.get("Curiosity", 0.5)
        survival = self.free_will.vectors.get("Survival", 0.3)
        
        #                    
        if survival > curiosity + 0.2:
            logger.info("  Decision: DEFER (survival > curiosity)")
            return ExplorationDecision.DEFER
        
        #            
        if curiosity > 0.6:
            logger.info("  Decision: EXPLORE (high curiosity)")
            return ExplorationDecision.EXPLORE
        
        #              
        if curiosity > 0.4:
            logger.info("  Decision: ASK_HUMAN (moderate curiosity)")
            return ExplorationDecision.ASK_HUMAN
        
        #        
        logger.info("  Decision: SKIP (low curiosity)")
        return ExplorationDecision.SKIP
    
    def execute_exploration(self, need: ExplorationNeed) -> ExplorationResult:
        """
                 +       +      
        
          :
        1.          (ExplorationCore)
        2.        "       ?"   
        3.          (Wikipedia, InnerDialogue, Human)
        4.           
        """
        logger.info(f"  Executing exploration: {need.question[:50]}...")
        
        answer = None
        principle = None
        source = "internal"
        attempted_methods = []
        failure_reasons = []
        
        # ===    1: ExplorationCore (     ) ===
        attempted_methods.append("exploration_core")
        if self.exploration_core:
            try:
                result = self.exploration_core.explore(need.question)
                if result:
                    answer = str(result)[:500]
                    source = "external_file"
                    logger.info("     Method 1 (ExplorationCore): SUCCESS")
            except Exception as e:
                failure_reasons.append(f"ExplorationCore: {str(e)[:50]}")
                logger.info(f"     Method 1 (ExplorationCore): FAILED - {str(e)[:30]}")
        else:
            failure_reasons.append("ExplorationCore: not connected")
        
        # ===            ===
        if not answer:
            logger.info("     Primary method failed. Trying alternatives...")
            
            # ===    2: Wikipedia API       ===
            attempted_methods.append("wikipedia_api")
            answer, wiki_reason = self._try_wikipedia(need.question)
            if answer:
                source = "wikipedia"
                logger.info("     Method 2 (Wikipedia): SUCCESS")
            else:
                failure_reasons.append(f"Wikipedia: {wiki_reason}")
                logger.info(f"     Method 2 (Wikipedia): FAILED - {wiki_reason[:30]}")
        
        # ===         3:       ===
        if not answer:
            attempted_methods.append("inner_dialogue")
            answer, inner_reason = self._try_inner_dialogue(need.question)
            if answer:
                source = "inner_dialogue"
                logger.info("     Method 3 (InnerDialogue): SUCCESS")
            else:
                failure_reasons.append(f"InnerDialogue: {inner_reason}")
                logger.info(f"     Method 3 (InnerDialogue): FAILED - {inner_reason[:30]}")
        
        # ===           :            ===
        if not answer:
            failure_analysis = self._analyze_failure(need, failure_reasons)
            logger.info(f"     All methods failed. Analysis: {failure_analysis['reason']}")
            
            #      
            if failure_analysis["suggested_action"] == "ask_human":
                source = "pending_human"
                #               
                answer = None
            elif failure_analysis["suggested_action"] == "defer":
                source = "deferred"
            elif failure_analysis["suggested_action"] == "decompose":
                #                
                decomposed = self._decompose_question(need.question)
                if decomposed:
                    logger.info(f"     Decomposing question into {len(decomposed)} sub-questions")
                    #                  (          1  )
                    sub_result = self._try_wikipedia(decomposed[0])
                    if sub_result[0]:
                        answer = sub_result[0]
                        source = "decomposed_wikipedia"
        
        # ===     :          ===
        if answer:
            # AutonomousLearner    
            if self.learner:
                try:
                    learn_result = self.learner.experience(
                        content=f"Q: {need.question}\nA: {answer}",
                        subject=need.question[:30],
                        domain=need.domain
                    )
                    if learn_result.get("learned_concept"):
                        principle = learn_result["learned_concept"]
                        logger.info(f"     Learned: {principle}")
                except Exception as e:
                    logger.debug(f"   AutonomousLearner failed: {e}")
            
            # WhyEngine      
            if self.why_engine:
                try:
                    crystallize = self.why_engine.analyze(
                        subject="crystallization",
                        content=f"  : {need.question}\n : {answer}",
                        domain=need.domain
                    )
                    if "[     ]" not in crystallize.underlying_principle:
                        principle = crystallize.underlying_principle
                        logger.info(f"     Crystallized: {principle[:60]}...")
                except Exception as e:
                    logger.debug(f"   Crystallization failed: {e}")
        
        result = ExplorationResult(
            question=need.question,
            answer=answer,
            principle_extracted=principle,
            source=source,
            success=answer is not None
        )
        
        self.exploration_history.append(result)
        return result
    
    def _try_wikipedia(self, question: str) -> tuple:
        """Wikipedia API          """
        try:
            import urllib.request
            import json
            
            #           (주권적 자아)
            keywords = question.replace("?", "").replace("  ", "").replace("    ", "").strip()
            keywords = keywords.split()[-1] if keywords.split() else question[:10]
            
            url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(keywords)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                extract = data.get('extract', '')
                if extract and len(extract) > 50:
                    return (extract[:500], None)
                else:
                    return (None, "No sufficient content")
        except Exception as e:
            return (None, str(e)[:50])
    
    def _try_inner_dialogue(self, question: str) -> tuple:
        """               """
        try:
            from Core.L5_Mental.M1_Cognition.Consciousness.Consciousness.inner_dialogue import DeepContemplation
            dc = DeepContemplation(max_depth=2)
            result = dc.dive(question)
            
            if result.get("final_principle") and "[     ]" not in result["final_principle"]:
                return (result["final_principle"], None)
            else:
                return (None, "Only reached unknown territory")
        except Exception as e:
            return (None, str(e)[:50])
    
    def _try_naver(self, question: str) -> tuple:
        """          (주권적 자아)"""
        if not self.naver or not self.naver.available:
            return (None, "Naver not available")
        
        try:
            result = self.naver.search_best(question)
            
            if result["success"] and result["results"]:
                #           
                first = result["results"][0]
                content = f"{first['title']}: {first['description']}"
                return (content, None)
            else:
                return (None, "No Naver results")
        except Exception as e:
            return (None, str(e)[:50])
    
    def _try_with_english_translation(self, question: str) -> tuple:
        """
                               
        
         : "  "   "freedom"   Wikipedia   
        """
        if not self.lang_mapper:
            return (None, "No language mapper")
        
        #              
        words = question.replace("?", "").replace("  ", " ").replace("    ", "").split()
        
        for word in words:
            #              
            english = self.lang_mapper.get_english(word)
            
            #       (코드 베이스 구조 로터)
            if english and english != word:
                logger.info(f"     Trying English: {word}   {english}")
                
                # Wikipedia       
                wiki_result, wiki_error = self._try_wikipedia(english)
                if wiki_result:
                    return (wiki_result, None)
        
        return (None, "No English translation available")
    
    def _analyze_failure(self, need: ExplorationNeed, reasons: List[str]) -> Dict[str, Any]:
        """
                        
        
             : "           ?"
        """
        analysis = {
            "question": need.question,
            "attempted_methods": len(reasons),
            "reasons": reasons,
            "reason": "unknown",
            "suggested_action": "ask_human"
        }
        
        #         
        all_reasons = " ".join(reasons).lower()
        
        if "not connected" in all_reasons or "not available" in all_reasons:
            analysis["reason"] = "         "
            analysis["suggested_action"] = "defer"  #             
            
        elif "timeout" in all_reasons or "connection" in all_reasons:
            analysis["reason"] = "       "
            analysis["suggested_action"] = "defer"
            
        elif "not found" in all_reasons or "no content" in all_reasons:
            analysis["reason"] = "      -           "
            analysis["suggested_action"] = "decompose"  #      
            
        elif "unknown" in all_reasons or "     " in all_reasons:
            analysis["reason"] = "      "
            analysis["suggested_action"] = "ask_human"  #         
        
        else:
            analysis["reason"] = "     "
            analysis["suggested_action"] = "ask_human"
        
        return analysis
    
    def _decompose_question(self, question: str) -> List[str]:
        """
                       
        
         : "         ?"   ["  ", "  ", "  "]
        """
        #          
        sub_questions = []
        
        #         
        core_word = question.replace("?", "").replace("  ", "").replace("    ", "").strip()
        
        if core_word:
            sub_questions.append(core_word)
            #         
            related = {
                "  ": ["  ", "  ", "  "],
                "  ": ["  ", "  ", "  "],
                "  ": ["  ", "  ", "  "],
            }
            if core_word in related:
                sub_questions.extend(related[core_word])
        
        return sub_questions
    
    def explore_all_sources(self, question: str) -> List[SourceQuality]:
        """
                              
        
        "                ,                   "
        
            : Naver > Wikipedia > InnerDialogue (자기 성찰 엔진)
        """
        logger.info(f"  Exploring ALL sources for: {question[:40]}...")
        
        sources = []
        
        # 1. Naver (       -      )
        naver_content, naver_error = self._try_naver(question)
        if naver_content:
            quality = self._evaluate_source_quality(question, naver_content, "naver")
            sources.append(quality)
            logger.info(f"     Naver: quality={quality.quality_score:.2f}")
        
        # 2. Wikipedia
        wiki_content, wiki_error = self._try_wikipedia(question)
        if wiki_content:
            quality = self._evaluate_source_quality(question, wiki_content, "wikipedia")
            sources.append(quality)
            logger.info(f"     Wikipedia: quality={quality.quality_score:.2f}")
        
        # 3. InnerDialogue  
        inner_content, inner_error = self._try_inner_dialogue(question)
        if inner_content:
            quality = self._evaluate_source_quality(question, inner_content, "inner_dialogue")
            sources.append(quality)
            logger.info(f"     InnerDialogue: quality={quality.quality_score:.2f}")
        
        # 3. ExplorationCore (     )
        if self.exploration_core:
            try:
                result = self.exploration_core.explore(question)
                if result:
                    content = str(result)[:500]
                    quality = self._evaluate_source_quality(question, content, "file_based")
                    sources.append(quality)
                    logger.info(f"     ExplorationCore: quality={quality.quality_score:.2f}")
            except:
                pass
        
        # 4. Naver (         )
        naver_content, naver_error = self._try_naver(question)
        if naver_content:
            quality = self._evaluate_source_quality(question, naver_content, "naver")
            sources.append(quality)
            logger.info(f"     Naver: quality={quality.quality_score:.2f}")
        
        # 5. Wikipedia             
        if not wiki_content:
            english_content, english_error = self._try_with_english_translation(question)
            if english_content:
                quality = self._evaluate_source_quality(question, english_content, "wikipedia_en")
                sources.append(quality)
                logger.info(f"     Wikipedia (English): quality={quality.quality_score:.2f}")
        
        logger.info(f"     Total sources found: {len(sources)}")
        return sources
    
    def _evaluate_source_quality(self, question: str, content: str, source_name: str) -> SourceQuality:
        """
                
        
             :
        - reliability:              
        - relevance:          (주권적 자아)
        - depth:        (   +   )
        """
        #        (   )
        reliability_map = {
            "wikipedia": 0.8,       #       
            "inner_dialogue": 0.5,  #    (     )
            "file_based": 0.6,      #   
            "human": 1.0,           #    (     )
        }
        reliability = reliability_map.get(source_name, 0.5)
        
        #     (                  )
        question_words = set(question.replace("?", "").split())
        content_words = set(content.split())
        overlap = len(question_words & content_words)
        relevance = min(1.0, overlap / max(len(question_words), 1) * 2)
        
        #    (      +     )
        sentence_count = content.count(".") + content.count(" ") + 1
        length_score = min(1.0, len(content) / 500)  # 500    
        structure_score = min(1.0, sentence_count / 5)  # 5     
        depth = (length_score + structure_score) / 2
        
        #      
        quality_score = (reliability * 0.4) + (relevance * 0.3) + (depth * 0.3)
        
        return SourceQuality(
            source_name=source_name,
            content=content,
            quality_score=quality_score,
            reliability=reliability,
            relevance=relevance,
            depth=depth
        )
    
    def select_best_source(self, sources: List[SourceQuality]) -> Optional[SourceQuality]:
        """
                   
        
            quality_score           ,
                           
        """
        if not sources:
            return None
        
        #                 
        best = max(sources, key=lambda s: s.quality_score)
        
        logger.info(f"     Best source: {best.source_name} (score={best.quality_score:.2f})")
        return best
    
    def explore_with_best_source(self, question: str) -> Optional[ExplorationResult]:
        """
                                
        
        "                   "
        """
        #         
        sources = self.explore_all_sources(question)
        
        if not sources:
            logger.info("     No sources succeeded")
            return ExplorationResult(
                question=question,
                answer=None,
                principle_extracted=None,
                source="none",
                success=False
            )
        
        #      
        best = self.select_best_source(sources)
        
        #        (WhyEngine)
        principle = None
        if self.why_engine and best.content:
            try:
                crystallize = self.why_engine.analyze(
                    subject="crystallization",
                    content=f"  : {question}\n : {best.content}",
                    domain="general"
                )
                if "[     ]" not in crystallize.underlying_principle:
                    principle = crystallize.underlying_principle
            except:
                pass
        
        #            (                  )
        if self.potential_store and best.content:
            #           
            subject = question.replace("?", "").replace("  ", "").replace("    ", "").strip()
            
            #            (frequency=0.3   )
            pk = self.potential_store.store(
                subject=subject,
                definition=best.content[:200],  #   
                source=best.source_name
            )
            
            #          (              )
            self.potential_store.auto_connect(subject)
            
            logger.info(f"     Stored as potential: {subject} (freq={pk.frequency:.2f})")
            
            #            
            if pk.is_crystallizable():
                crystallized = self.potential_store.crystallize(subject)
                if crystallized:
                    principle = f"{crystallized['concept']}: {crystallized['definition'][:100]}"
                    logger.info(f"     Crystallized: {subject}")
        
        return ExplorationResult(
            question=question,
            answer=best.content,
            principle_extracted=principle,
            source=best.source_name,
            success=True
        )
    
    def process_exploration_need(self, content: str, subject: str = "unknown") -> Optional[ExplorationResult]:
        """
                   
        
        1.         
        2.       
        3.       
        4.      
        """
        # 1.   
        need = self.detect_exploration_need(content, subject)
        if not need:
            return None
        
        # 2.       
        self.stimulate_curiosity(need)
        
        # 3.   
        decision = self.decide_exploration(need)
        
        # 4.          
        if decision == ExplorationDecision.EXPLORE:
            return self.execute_exploration(need)
        
        elif decision == ExplorationDecision.ASK_HUMAN:
            #                 
            logger.info(f"     Pending question for human: {need.question}")
            return ExplorationResult(
                question=need.question,
                answer=None,
                principle_extracted=None,
                source="pending_human",
                success=False
            )
        
        elif decision == ExplorationDecision.DEFER:
            #             
            logger.info(f"     Deferred for later")
            return None
        
        else:  # SKIP
            return None
    
    def get_pending_explorations(self) -> List[ExplorationNeed]:
        """           """
        return self.exploration_queue
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """     """
        successful = [r for r in self.exploration_history if r.success]
        return {
            "total_explorations": len(self.exploration_history),
            "successful": len(successful),
            "pending": len(self.exploration_queue),
            "principles_extracted": len([r for r in self.exploration_history if r.principle_extracted])
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Exploration Bridge Demo")
    print("   '[     ]'        ")
    print("=" * 60)
    
    bridge = ExplorationBridge()
    
    #     1:              
    print("\n  Test: Exploration flow")
    result = bridge.process_exploration_need("         ?", "love")
    
    if result:
        print(f"   Success: {result.success}")
        print(f"   Source: {result.source}")
        print(f"   Principle: {result.principle_extracted}")
    else:
        print("   No exploration executed")
    
    #   
    stats = bridge.get_exploration_stats()
    print(f"\n  Stats: {stats}")
    
    print("\n" + "=" * 60)
    print("  Demo complete!")
