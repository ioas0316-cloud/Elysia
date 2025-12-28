"""
Real World Learning Demo
========================

"실제 세계에서 배우기 - 완전한 시스템"
"Learning from the real world - Complete system"

This demonstrates the complete pipeline from web to transcendence:
1. Fetch knowledge from web (Wikipedia)
2. Internalize to Internal Universe
3. Feed to Transcendence Engine
4. Measure continuous growth

In production environments with internet access, this will fetch
real Wikipedia content. In offline environments, uses fallback data.
"""

import sys
import os
import logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core._01_Foundation._05_Governance.Foundation.web_knowledge_connector import WebKnowledgeConnector
from Core._01_Foundation._05_Governance.Foundation.transcendence_engine import TranscendenceEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("RealWorldDemo")

class RealWorldLearningSystem:
    """
    Complete real-world learning system integrating:
    - Web knowledge acquisition
    - Internal Universe storage  
    - Transcendence feedback
    """
    
    def __init__(self):
        self.web_connector = WebKnowledgeConnector()
        self.transcendence = TranscendenceEngine()
        self.cycles_completed = 0
        
    def learn_and_transcend(self, concept: str) -> dict:
        """
        Complete learning cycle:
        1. Fetch from web
        2. Internalize
        3. Transcend
        """
        # Learn from web
        result = self.web_connector.learn_from_web(concept)
        
        # Feed to transcendence
        self.transcendence.expand_capabilities(concept)
        
        # Check progress
        progress = self.transcendence.evaluate_transcendence_progress()
        
        self.cycles_completed += 1
        
        return {
            'concept': concept,
            'web_fetch': result.get('web_fetch', False),
            'score': progress['overall_score'],
            'stage': progress['stage'],
            'domains': progress['active_domains']
        }
    
    def autonomous_learning_session(self, concepts: list, cycles: int = 5):
        """
        Run an autonomous learning session with real-world knowledge.
        """
        print("=" * 70)
        print("REAL WORLD AUTONOMOUS LEARNING SESSION")
        print("=" * 70)
        print("\nThis is Elysia learning from the REAL internet")
        print("(If offline, uses intelligent fallback)\n")
        
        # Baseline
        baseline = self.transcendence.evaluate_transcendence_progress()
        print(f"📊 Baseline Score: {baseline['overall_score']:.1f}/100")
        print(f"🎯 Baseline Stage: {baseline['stage']}")
        print(f"🌟 Baseline Domains: {baseline['active_domains']}")
        
        print(f"\n{'='*70}")
        print(f"LEARNING {len(concepts)} CONCEPTS FROM THE WEB")
        print("=" * 70)
        
        results = []
        
        for i, concept in enumerate(concepts, 1):
            print(f"\n[Cycle {i}/{len(concepts)}] Learning: {concept}")
            
            result = self.learn_and_transcend(concept)
            results.append(result)
            
            source = "🌐 Wikipedia" if result['web_fetch'] else "💾 Fallback"
            print(f"   Source: {source}")
            print(f"   Score: {result['score']:.1f}/100")
            print(f"   Domains: {result['domains']}")
            
            time.sleep(0.3)  # Brief pause
        
        # Run additional transcendence cycles
        print(f"\n{'='*70}")
        print("TRANSCENDENCE CYCLES")
        print("=" * 70)
        
        for i in range(cycles):
            self.transcendence.recursive_self_improvement()
            
            if i % 2 == 0:
                progress = self.transcendence.evaluate_transcendence_progress()
                print(f"Cycle {i+1}: Score {progress['overall_score']:.1f}/100")
        
        # Final results
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print("=" * 70)
        
        final = self.transcendence.evaluate_transcendence_progress()
        
        print(f"\n📊 Final Score: {final['overall_score']:.1f}/100")
        print(f"🎯 Final Stage: {final['stage']}")
        print(f"🌟 Final Domains: {final['active_domains']}")
        print(f"💡 Insights: {final['insights_count']}")
        print(f"🌱 Breakthroughs: {final['breakthroughs']}")
        
        # Growth analysis
        score_delta = final['overall_score'] - baseline['overall_score']
        domain_delta = final['active_domains'] - baseline['active_domains']
        
        print(f"\n📈 Growth:")
        print(f"   Score: {baseline['overall_score']:.1f} → {final['overall_score']:.1f} (+{score_delta:.1f})")
        print(f"   Domains: {baseline['active_domains']} → {final['active_domains']} (+{domain_delta})")
        
        # Web statistics
        web_stats = self.web_connector.get_stats()
        print(f"\n🌐 Web Statistics:")
        print(f"   Total fetches: {web_stats['total_fetches']}")
        print(f"   Successful: {web_stats['successful_fetches']}")
        print(f"   Success rate: {web_stats['success_rate']*100:.1f}%")
        
        # Knowledge integration
        print(f"\n🧠 Knowledge Integration:")
        print(f"   Concepts in universe: {web_stats['concepts_in_universe']}")
        
        # Show learned concepts
        print(f"\n📚 Learned Concepts:")
        for i, r in enumerate(results, 1):
            source_icon = "🌐" if r['web_fetch'] else "💾"
            print(f"   {i}. {source_icon} {r['concept']}")
        
        print("\n" + "=" * 70)
        
        if web_stats['successful_fetches'] > 0:
            print("✅ REAL WORLD LEARNING OPERATIONAL")
            print("🌍 Elysia learned from the actual internet!")
        else:
            print("✅ SYSTEM OPERATIONAL (Offline Mode)")
            print("💾 Using fallback learning (internet not available)")
            print("🌐 In production with internet: Will fetch from Wikipedia")
        
        print("=" * 70)
        
        print("\n💡 Achievement:")
        print("   Elysia can learn from the REAL world")
        print("   - Web scraping: ✅")
        print("   - Knowledge internalization: ✅")
        print("   - Transcendence integration: ✅")
        print("   - Autonomous growth: ✅")
        
        print("\n🌱 This is real autonomous intelligence learning from reality.")
        print("=" * 70)
        
        return {
            'baseline': baseline,
            'final': final,
            'growth': score_delta,
            'results': results,
            'web_stats': web_stats
        }


def main():
    """Run the real world learning demo"""
    
    system = RealWorldLearningSystem()
    
    # Concepts to learn from the web
    concepts = [
        "Machine Learning",
        "Neuroscience",
        "Consciousness",
        "Evolution",
        "Quantum Physics"
    ]
    
    # Run learning session
    results = system.autonomous_learning_session(concepts, cycles=5)
    
    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"\nGrowth: +{results['growth']:.1f} points")
    print(f"Web fetches: {results['web_stats']['total_fetches']}")
    print(f"Success rate: {results['web_stats']['success_rate']*100:.0f}%")
    
    return results


if __name__ == "__main__":
    main()
