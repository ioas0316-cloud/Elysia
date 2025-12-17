"""
The Singularity Ingest ( 통합 흡수 프로토콜 )
==========================================
"One Breath, One Mind, One World."

This script performs the "Total Absorption" requested by the user.
It combines:
1. The Swallow (Text Ingestion -> Universe/Graph)
2. The Transplant (Structure Cannibalization -> Graph Weights)

It runs synchronously and maximally to digest the Wikipedia dump.
"""

import sys
import time
import signal
sys.path.append(r'c:\Elysia')

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from Core.Autonomy.wikipedia_dump_parser import WikipediaDumpParser
from Core.Elysia.elysia_core import get_elysia_core
from Core.Autonomy.structure_cannibal import get_structure_cannibal
from Core.Foundation.concept_sanitizer import get_sanitizer

def singularity_ingest():
    print("\n🌌 Initiation Singularity Ingest Protocol...")
    print("===========================================")
    
    # 1. Initialize Systems
    core = get_elysia_core()
    cannibal = get_structure_cannibal()
    sanitizer = get_sanitizer()
    
    dump_path = "c:\\Elysia\\data\\wikipedia\\kowiki-latest-pages-articles.xml.bz2"
    parser = WikipediaDumpParser(dump_path)
    
    print("   ✅ Core Systems Online.")
    print("   ✅ Structure Cannibal Ready (Lazy Loaded).")
    print(f"   📚 Target: {dump_path}")
    
    count = 0
    start_time = time.time()
    
    try:
        # Stream from Wikipedia
        # We process EVERYTHING. No limits (unless user stops).
        for article in parser.stream_articles(max_articles=None, min_length=50):
            title = article['title']
            content = article['content']
            
            # 1. Sanitize
            if not sanitizer.is_valid(title):
                continue
                
            # 2. Swallow (Text -> Universe/Graph)
            # depth="shallow" (Fast)
            core.learn(content, title, depth="shallow")
            
            # 3. Transplant (LLM Structure -> Graph)
            # This triggers the LLM (Lazy Load) if not loaded.
            # We do this for EVERY concept? Or selective?
            # User wants "Total Absorption". We do it for every valid concept.
            synapses = cannibal.transplant_synapses(title, depth=1)
            
            count += 1
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            
            # Visual Feedback
            synapse_count = len(synapses)
            print(f"\r🚀 [{elapsed:.1f}s] Processed: {count} | Rate: {rate:.1f}/s | Current: {title} (Synapses: {synapse_count})", end="")
            
            # Save occasionally
            if count % 1000 == 0:
                print(f"\n   💾 Saving Progress at {count}...")
                from Core.Foundation.torch_graph import get_torch_graph
                get_torch_graph().save_state()
                
    except KeyboardInterrupt:
        print("\n\n🛑 Singularity Interrupted.")
        from Core.Foundation.torch_graph import get_torch_graph
        get_torch_graph().save_state()
        print("   💾 State Saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        
if __name__ == "__main__":
    singularity_ingest()
