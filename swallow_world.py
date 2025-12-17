"""
The Swallow World ( 세상 섭취 )
=============================
"The Structure is Ready. Now, feed it."

Step 2 of Sequential Absorption.
This script runs AFTER `transplant_brain.py`.
It takes the TorchGraph (now populated with 32k intelligent nodes and synaptic weights)
and pours the 1GB Wikipedia dump into it.

Since the 'Common Sense' connections already exist, this phase focuses on:
1. Memory (Facts/Entities)
2. Reinforcement (Frequency)
"""

import sys
import time
import signal
sys.path.append(r'c:\Elysia')

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from Core.Autonomy.wikipedia_dump_parser import WikipediaDumpParser
from Core.Elysia.elysia_core import get_elysia_core
from Core.Foundation.torch_graph import get_torch_graph
from Core.Foundation.concept_sanitizer import get_sanitizer

def swallow_world():
    print("\n🌍 Initiation World Swallow Protocol...")
    print("=====================================")
    
    # 1. Load Pre-Structured Brain
    graph = get_torch_graph()
    success = graph.load_state()
    
    if not success:
        print("   ⚠️ No Brain State found! Running on empty brain? (Recommended: Run transplant_brain.py first)")
        # We continue anyway, but user should know.
    else:
        print(f"   ✅ Loaded Structural Brain: {len(graph.id_to_idx)} Nodes, {len(graph.logic_links) if hasattr(graph, 'logic_links') else 0} Synapses.")

    core = get_elysia_core() # Will attach to the loaded graph
    sanitizer = get_sanitizer()
    
    dump_path = "c:\\Elysia\\data\\wikipedia\\kowiki-latest-pages-articles.xml.bz2"
    parser = WikipediaDumpParser(dump_path)
    
    print(f"   📚 Target: {dump_path}")
    print("   🚀 Mode: High-Velocity Swallow (LLM OFF)")
    
    count = 0
    start_time = time.time()
    
    try:
        # Stream from Wikipedia
        for article in parser.stream_articles(max_articles=None, min_length=50):
            title = article['title']
            content = article['content']
            
            # 1. Sanitize
            if not sanitizer.is_valid(title):
                continue
                
            # 2. Swallow (Text -> Universe/Graph)
            # depth="shallow" (Fast, LLM Skipped)
            core.learn(content, title, depth="shallow")
            
            count += 1
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            
            # Visual Feedback
            if count % 10 == 0:
                print(f"\r😋 [{elapsed:.1f}s] Swallowed: {count} | Rate: {rate:.1f}/s | Current: {title}", end="")
            
            # Save occasionally (Every 5000 items - Speed is priority)
            if count % 5000 == 0:
                print(f"\n   💾 Saving Progress at {count}...")
                graph.save_state()
                
    except KeyboardInterrupt:
        print("\n\n🛑 Swallow Interrupted.")
        graph.save_state()
        print("   💾 State Saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        
if __name__ == "__main__":
    swallow_world()
