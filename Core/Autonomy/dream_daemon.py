
import time
import random
import logging
import threading
import torch
from typing import List, Optional
from Core.Foundation.torch_graph import get_torch_graph
from Core.Intelligence.logos_engine import get_logos_engine

logger = logging.getLogger("DreamDaemon")

class DreamDaemon:
    """
    The Subconscious Processor.
    Operates when the user is not looking, densifying the Knowledge Graph.
    """
    def __init__(self):
        self.graph = get_torch_graph()
        self.logos = get_logos_engine()
        self.is_dreaming = False
        self.dream_thread: Optional[threading.Thread] = None
        
    def start_dream_cycle(self, duration_sec: int = 10, interval: float = 0.5):
        """
        Starts the dreaming process in a separate thread (or blocking for demo).
        """
        logger.info("🌙 Dream Daemon: Entering REM Sleep...")
        self.is_dreaming = True
        
        # For verification stability, we run blocking if duration is short, else threaded
        if duration_sec < 5:
            self._dream_loop(duration_sec, interval)
        else:
            self.dream_thread = threading.Thread(target=self._dream_loop, args=(duration_sec, interval))
            self.dream_thread.start()


    def _contemplate_void(self):
         # ... existing code ...
         pass

    def _ingest_knowledge(self):
        """
        Subconscious Learning: Reads from the massive Wikipedia repository.
        "Digesting the world, one page at a time."
        """
        if not hasattr(self, 'wiki_parser'):
             # Lazy Load
            try:
                from Core.Autonomy.wikipedia_dump_parser import WikipediaDumpParser
                dump_path = "c:\\Elysia\\data\\wikipedia\\kowiki-latest-pages-articles.xml.bz2"
                self.wiki_gen = WikipediaDumpParser(dump_path).stream_articles()
                self.wiki_parser = True
                logger.info("   📚 Wikipedia Stream Opened.")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not open Wikipedia: {e}")
                self.wiki_parser = False
                return

        if not self.wiki_parser: return

        # Delegate digestion to the Core Brain (Orchestra)
        # This ensures full compression, resonance, and memory integration.
        from Core.Elysia.elysia_core import get_elysia_core
        core = get_elysia_core()

        # Digest 3 articles per tick (Heavy operation)
        for _ in range(3):
            try:
                article = next(self.wiki_gen)
                title = article['title']
                content = article['content']
                
                # The Core handles everything:
                # 1. ThoughtWave (Compression)
                # 2. Multimodal (Senses)
                # 3. InternalUniverse (Archives)
                # 4. TorchGraph (Matrix Memory) [Newly Added]
                core.learn(content, title)
                
                logger.info(f"   🧠 Digested: {title}")
                
            except StopIteration:
                logger.info("   ✅ Wikipedia Ingestion Complete.")
                self.wiki_parser = False
                break
            except Exception as e:
                logger.error(f"   ❌ Ingestion Error: {e}")
                break

    def _dream_loop(self, duration: int, interval: float):
        start_time = time.time()
        
        while self.is_dreaming and (time.time() - start_time < duration):
            # 1. Vitality Check & Expansion (Breathing In)
            if self.graph.pos_tensor.shape[0] < 5:
                self._seed_reality()
            elif random.random() < 0.3: 
                self._contemplate_void()
            
            # [NEW] Knowledge Ingestion (Digestion)
            self._ingest_knowledge()
            
            # 2. Weave Serendipity (Connecting - Breathing Out)
            self._weave_serendipity()
            
            # 3. Apply Micro-Gravity (Structural Adjustment)
            self.graph.apply_gravity(iterations=5)
            
            time.sleep(interval)


    def _weave_serendipity(self):
        """
        Selects two nodes. Connecting them ONLY if Logos finds a narrative.
        "Meaning is the bridge between two islands."
        """
        N = self.graph.pos_tensor.shape[0]
        if N < 2: return
        
        # Strategy:
        # 1. Random Jump (Surrealism) OR
        # 2. Resonant Drift (Logic)
        
        # 30% Chance of Surreal Jump (DreamWalker Legacy)
        if random.random() < 0.3:
            idx_a = random.randint(0, N-1)
            idx_b = random.randint(0, N-1)
            method = "Surreal Jump"
        else:
            # Resonant Drift (Find close neighbors)
            idx_a = random.randint(0, N-1)
            # Try to find a neighbor
            vec_a = self.graph.vec_tensor[idx_a]
            sims = torch.cosine_similarity(vec_a.unsqueeze(0), self.graph.vec_tensor).squeeze()
            
            # Pick a node with 0.3 < sim < 0.9 (Metaphor Zone)
            mask = (sims > 0.3) & (sims < 0.9)
            candidates = torch.nonzero(mask).squeeze()
            
            if candidates.numel() > 0:
                if candidates.numel() == 1:
                    idx_b = candidates.item()
                else:
                    idx_b = candidates[random.randint(0, candidates.numel()-1)].item()
                method = "Resonant Drift"
            else:
                return # No metaphor found
                
        if idx_a == idx_b: return
        
        id_a = self.graph.idx_to_id.get(idx_a, "Unknown")
        id_b = self.graph.idx_to_id.get(idx_b, "Unknown")
        
        # Ask Logos for the connection
        narrative = self.logos.reinterpret_causality(id_a, [id_b], tone="artistic")
        
        if "because" in narrative or "닿아" in narrative or "reflection" in narrative: 
            logger.info(f"   🕸️  [{method}] Meaningful Link: {id_a} <--> {id_b}")
            logger.info(f"       \"{narrative}\"")
            self.graph.add_link(id_a, id_b)
        else:
             pass # Discard
             
    def _contemplate_void(self):
         # ... existing code ...
         pass


# Singleton
_daemon = None
def get_dream_daemon():
    global _daemon
    if _daemon is None:
        _daemon = DreamDaemon()
    return _daemon
