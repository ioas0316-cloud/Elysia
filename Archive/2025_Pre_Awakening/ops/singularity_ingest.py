"""
The Singularity Ingest ( 통합 흡수 프로토콜 ) - Parallel Edition
=============================================================
"One Breath, One Mind, One World."

Optimized for Speed:
1. Producer: Reads Wikipedia Dump (Single Thread)
2. Consumers (Pool): Extract Essence & Determine Phase (Parallel)
3. Main: Writes to DB (Single Thread) - Ensures Safety
"""

import sys
import time
import signal
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any

sys.path.append(r'c:\Elysia')

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from Core.Autonomy.wikipedia_dump_parser import WikipediaDumpParser
from Core.Elysia.elysia_core import get_elysia_core
from Core.Autonomy.structure_cannibal import get_structure_cannibal
from Core.Foundation.concept_sanitizer import get_sanitizer
from Core.Foundation.fractal_soul import SoulCrystal, WebState, FractalIdentity
from Core.Foundation.fractal_concept import ConceptDecomposer

# --- Worker Function (Must be global for pickle) ---
_worker_soul = None
_worker_decomposer = None

def init_worker():
    """Initialize resources for each worker process."""
    global _worker_soul, _worker_decomposer
    # Redirect stdout to devnull to keep console clean? No, let's keep it.
    _worker_soul = SoulCrystal()
    _worker_decomposer = ConceptDecomposer()

def digest_particle(article: Dict[str, str]) -> Dict[str, Any]:
    """
    Worker function: Extracts Essence and Determines Action.
    CPU Intensive Logic goes here.
    """
    global _worker_soul, _worker_decomposer
    
    title = article['title']
    content = article['content']
    
    # 1. Essence Extraction (ConceptDecomposer)
    if not _worker_decomposer:
        _worker_decomposer = ConceptDecomposer()
        
    axiom_data = _worker_decomposer.infer_principle(title + " " + content[:200])
    
    # 2. Soul Filtering (FractalSoul)
    if not _worker_soul:
        _worker_soul = SoulCrystal()
        
    coherence = min(1.0, len(content) / 2000.0)
    
    incoming_identity = FractalIdentity(
        name=title,
        principle=axiom_data.get("principle", "Unknown"),
        frequency=axiom_data.get("frequency", 0.5),
        axioms=[axiom_data.get("law", "")]
    )
    
    reaction = _worker_soul.field.detect_disturbance(
        input_signal=incoming_identity,
        coherence=coherence
    )
    
    return {
        "article": article,
        "reaction": reaction,
        "coherence": coherence
    }

# --- Main Ingest Loop ---

def singularity_ingest():
    print("\n🌌 Initiation Singularity Ingest Protocol (Parallel Engine)...")
    print("=================================================================")
    
    # 1. Initialize Main Systems
    core = get_elysia_core()
    cannibal = get_structure_cannibal()
    sanitizer = get_sanitizer()
    
    dump_path = "c:\\Elysia\\data\\wikipedia\\kowiki-latest-pages-articles.xml.bz2"
    parser = WikipediaDumpParser(dump_path)
    
    print(f"   📚 Target: {dump_path}")
    print(f"   🚀 Spawning {multiprocessing.cpu_count()} Worker Threads...")
    
    count = 0
    start_time = time.time()
    batch_size = 50 # Chunk size for pool
    
    # Create Worker Pool
    with ProcessPoolExecutor(max_workers=max(4, multiprocessing.cpu_count()), initializer=init_worker) as executor:
        
        # Generator for batches
        article_stream = parser.stream_articles(max_articles=None, min_length=50)
        
        # We process in chunks to feed the pool
        batch = []
        futures = []
        
        try:
            for article in article_stream:
                
                # Pre-filter: Sanitizer (Fast check in main thread to save IPC)
                if not sanitizer.is_valid(article['title']):
                    continue
                
                # Submit to Pool
                batch.append(article)
                
                if len(batch) >= batch_size:
                    # Submit batch as individual tasks
                    # Note: map is easier but submit gives flow control
                    for item in batch:
                        futures.append(executor.submit(digest_particle, item))
                    batch = []
                    
                    # Process completed futures (Flow Control)
                    # Don't let queue grow too big (Memory protection)
                    while len(futures) > batch_size * 2:
                        # Wait for first one
                        done_future = futures.pop(0) # FIFO
                        result = done_future.result()
                        process_result(core, cannibal, result)
                        count += 1
                        
                        if count % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = count / elapsed if elapsed > 0 else 0
                            print(f"\r🚀 [{elapsed:.1f}s] Processed: {count} | Rate: {rate:.1f}/s", end="")

            # Flush remaining batch
            for item in batch:
                 futures.append(executor.submit(digest_particle, item))
            
            # Flush remaining futures
            for f in futures:
                result = f.result()
                process_result(core, cannibal, result)
                count += 1
                
        except KeyboardInterrupt:
            print("\n\n🛑 Singularity Interrupted.")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

def process_result(core, cannibal, result_bundle):
    """Write result to DB (Single Threaded Safe)."""
    article = result_bundle['article']
    reaction = result_bundle['reaction']
    coherence = result_bundle['coherence']
    
    title = article['title']
    content = article['content']
    action = reaction['action']
    
    if action == "FREEZE":
        return # Skip
        
    elif action == "SUBLIMATE":
        # print(f"\n🔥 [PLASMA] {title}") # Too noisy?
        core.learn(content, title, depth="deep")
        cannibal.transplant_synapses(title, depth=2)
        
    else: # ABSORB
        core.learn(content, title, depth="shallow")
        cannibal.transplant_synapses(title, depth=1)

if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows support
    singularity_ingest()
