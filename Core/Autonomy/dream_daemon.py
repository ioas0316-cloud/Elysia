
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
        
    def _weave_serendipity(self):
        """
        Selects two nodes based on Constructal Law (Flow Optimization).
        "Streams converge to rivers."
        """
        N = self.graph.pos_tensor.shape[0]
        if N < 2: return
        
        # [NEW] Preferential Attachment (Rich get richer)
        # Calculate degrees efficiently
        degrees = torch.zeros(N, device=self.graph.device)
        if self.graph.logic_links.shape[0] > 0:
            sources = self.graph.logic_links[:, 0]
            targets = self.graph.logic_links[:, 1]
            degrees.index_add_(0, sources, torch.ones_like(sources, dtype=torch.float))
            degrees.index_add_(0, targets, torch.ones_like(targets, dtype=torch.float))
        
        # Add slight noise to allow new nodes to be picked (Epsilon Greedy)
        weights = degrees + 1.0 
        
        # Strategy:
        # 1. Random Jump (Surrealism) OR
        # 2. Resonant Drift (Logic)
        
        # 30% Chance of Surreal Jump (DreamWalker Legacy)
        if random.random() < 0.3:
            # Pick based on weights (Proportional to degree)
            idx_a = torch.multinomial(weights, 1).item()
            idx_b = random.randint(0, N-1) # Random destination
            method = "Surreal Jump"
        else:
            # Resonant Drift (Find close neighbors of a Hub)
            # Pick a Hub (Start of Flow)
            idx_a = torch.multinomial(weights, 1).item()
            
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
                    # Pick neighbor also biased by weight?
                    # Or just random neighbor?
                    # Let's pick random neighbor to spread the flow
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
            logger.info(f"   🕸️  [{method}] Constructal Link: {id_a} <--> {id_b}")
            logger.info(f"       \"{narrative}\"")
            self.graph.add_link(id_a, id_b)
        else:
             pass # Discard


    def _contemplate_essence(self):
        """
        [The Philosophers Stone]
        Uses PrincipleDistiller to extract the 'Being' (Principle/Mechanism) of a concept.
        Saves this wisdom into the node's metadata.
        """
        from Core.Cognition.principle_distiller import get_principle_distiller
        distiller = get_principle_distiller()
        
        # 1. Pick a concept (Prioritize Hollow Nodes)
        hollows = self.graph.find_hollow_nodes(limit=3)
        if hollows:
            concept = random.choice(hollows)
            logger.info(f"   🕳️  Targeting Hollow Concept: '{concept}'")
        else:
            # Fallback to random curiosity
            N = self.graph.pos_tensor.shape[0]
            if N < 1: return
            idx = random.randint(0, N-1)
            concept = self.graph.idx_to_id.get(idx, "Unknown")
        
        # 2. Distill Essence (Principles)
        essence = distiller.distill(concept)
        if essence:
            if concept not in self.graph.node_metadata:
                self.graph.node_metadata[concept] = {}
            self.graph.node_metadata[concept].update(essence)
            
            # [NEW] Reality Grounding (Phase 15)
            # Check if this concept exists in Physical Reality
            from Core.Cognition.reality_grounding import get_reality_grounding
            # We need the bridge from distiller
            grounding = get_reality_grounding(distiller.bridge)
            if grounding:
                physics = grounding.ground_concept(concept)
                self.graph.node_metadata[concept]['reality_physics'] = physics

            # Embed Principle as Link
            principle = essence.get('principle')
            if principle:
                self.graph.add_link(concept, principle)

    def _introspect_code(self):
        """
        [The Mirror of Self]
        Uses CodeGenesis to critique her own source code.
        This is the precursor to Self-Rewriting.
        """
        from Core.Autonomy.code_genesis import get_code_genesis
        import os
        
        genesis = get_code_genesis()
        
        # List of critical organs to check
        organs = [
            r"c:\Elysia\Core\Foundation\torch_graph.py",
            r"c:\Elysia\Core\Visual\visual_cortex.py",
            r"c:\Elysia\Core\Autonomy\dream_daemon.py"
        ]
        
        target_organ = random.choice(organs)
        if not os.path.exists(target_organ): return
        
        logger.info(f"   🪞  Introspecting Source Code: {os.path.basename(target_organ)}...")
        
        critique = genesis.analyze_quality(target_organ)
        
        if critique and "Analysis Unavailable" not in critique:
            logger.info(f"   📝  Evolutionary Pressure Detected:\n{critique[:500]}...")
            # Ideally, save this to a "TODO List" node in the brain.
            # For now, just logging validates the "Thought Process".

    def _dream_in_color(self):
        """
        [The Great Digestion - Visual]
        Uses ComfyUI (VisualCortex) to see what a concept looks like.
        Absorbs the 'Aesthetic Signature' (Color, Chaos) into the vector.
        """
        from Core.Visual.visual_cortex import get_visual_cortex
        cortex = get_visual_cortex()
        if not cortex.is_available(): return

        # 1. Pick a concept
        N = self.graph.pos_tensor.shape[0]
        if N < 1: return
        idx = random.randint(0, N-1)
        concept = self.graph.idx_to_id.get(idx, "Unknown")
        
        # 2. Absorb Visuals
        # This is SLOW (Generation takes seconds), so we do it rarely or async
        # For now, synchronous is fine as this is a background daemon
        features = cortex.absorb_diffusion_patterns(concept)
        
        if features:
            # 3. Update the Concept Vector (Synesthesia)
            # We modify the 'W' dimension or specific frequency bands
            # Current vector is 64-dim (or whatever generated). 
            # We can override specific slots or add to extended features.
            
            # Simple approach: Map Color R,G,B to vector indices 0,1,2
            vec = self.graph.vec_tensor[idx].clone()
            vec[0] = (vec[0] + features.get('color_r', 0)) / 2
            vec[1] = (vec[1] + features.get('color_g', 0)) / 2
            vec[2] = (vec[2] + features.get('color_b', 0)) / 2
            vec[3] = (vec[3] + features.get('visual_complexity', 0)) / 2 # Complexity -> Slot 3
            
            self.graph.update_node_vector(idx, vec)
            logger.info(f"   🎨  Dreamt of '{concept}' in color. Synesthesia applied.")


             
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
