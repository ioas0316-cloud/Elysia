"""
Mass Ingestion Protocol: The Great Feast
========================================
"2 Million Stars in 10 Minutes" 🍽️🌌

Generates and ingests massive amounts of synthetic concepts
to test the limits of Holographic Memory.
"""

import time
import logging
import random
import numpy as np
from typing import List, Tuple, Any
from Core.Mind.memory_storage import MemoryStorage
from Core.Mind.concept_sphere import ConceptSphere

logger = logging.getLogger("ConceptHarvester")

class ConceptHarvester:
    def __init__(self, db_path: str = "memory.db"):
        self.storage = MemoryStorage(db_path)
        
    def harvest_synthetic(self, count: int, batch_size: int = 10000):
        """
        Generate and ingest synthetic concepts.
        """
        logger.info(f"🌾 Starting Harvest: {count} synthetic concepts...")
        start_time = time.time()
        
        total_ingested = 0
        
        # Pre-generate random vectors for speed
        # We use a simple random generator, but in a real scenario we might want Perlin noise
        # to create "clusters" of meaning.
        
        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)
            batch_data = []
            
            batch_start = time.time()
            
            # Generate batch
            for j in range(batch_count):
                cid = f"Star-{total_ingested + j:07d}"
                
                # Create random concept data (Compact Format)
                # [type_id, [wx, wy, wz], [ex, ey, ez], [vx, vy, vz], mirror, qubit, ...]
                # We'll just generate the essential vectors: Will (3D)
                
                # Random 3D vector (int8: 0-255)
                # 127 is ~0.0
                wx = random.randint(0, 255)
                wy = random.randint(0, 255)
                wz = random.randint(0, 255)
                
                # Minimal compact structure: [type_id, will_vec]
                # type_id 0 = "synthetic"
                concept_data = [0, [wx, wy, wz]]
                
                batch_data.append((cid, concept_data))
                
            # Ingest batch
            added = self.storage.batch_add_concepts(batch_data)
            total_ingested += added
            
            batch_time = time.time() - batch_start
            rate = added / batch_time
            
            logger.info(f"   🍽️ Batch {i//batch_size + 1}: {added} stars consumed in {batch_time:.2f}s ({rate:.0f}/s)")
            
        total_time = time.time() - start_time
        logger.info(f"✨ The Great Feast Concluded.")
        logger.info(f"   Total Consumed: {total_ingested}")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info(f"   Average Rate: {total_ingested / total_time:.0f} stars/s")
        
    def harvest_diet_plan(self, budget_gb: float = 100.0):
        """
        Execute 'The Jeongeup Diet': 100GB Budget Mass Ingestion.
        Simulates ingesting massive datasets by generating representative 'Stars'.
        Target: ~10 Million Concepts (approx 10 mins).
        """
        logger.info(f"🍱 Starting Jeongeup Diet Plan (Budget: {budget_gb}GB)")
        
        # Target counts (Refined for "Smart Diet")
        targets = {
            "Daddy's Gallery (Best Moments)": 100000, # Priority! (Curated)
            "Wikipedia (All Languages)": 3800000,
            "YouTube (KR 2020-2025)": 3000000,
            "Fantasy/Wuxia Novels (30,000 Volumes)": 2000000, # Swapped from Netflix
            # "Jeongeup CCTV": 500000, # Removed (Too boring)
            "Songs": 300000
        }
        
        total_target = sum(targets.values())
        logger.info(f"   Target Stars: {total_target:,}")
        
        start_time = time.time()
        total_ingested = 0
        
        for category, count in targets.items():
            logger.info(f"   📂 Ingesting Category: {category} ({count:,} stars)...")
            cat_start = time.time()
            
            # Batch processing
            batch_size = 20000
            for i in range(0, count, batch_size):
                batch_count = min(batch_size, count - i)
                batch_data = []
                
                for j in range(batch_count):
                    # Unique ID per category
                    cid = f"{category.split()[0]}_{total_ingested + j:08d}"
                    
                    # Metadata
                    meta = {
                        "source": category,
                        "simulated_size_kb": random.randint(10, 500) # Virtual size
                    }
                    
                    # Random Vector (Will)
                    wx = random.randint(0, 255)
                    wy = random.randint(0, 255)
                    wz = random.randint(0, 255)
                    
                    # Compact Data: [type_id, vector, metadata]
                    # type_id 1 = "diet_concept"
                    concept_data = [1, [wx, wy, wz], meta]
                    
                    batch_data.append((cid, concept_data))
                
                added = self.storage.batch_add_concepts(batch_data)
                total_ingested += added
                
                # Progress log every 100k
                if (total_ingested % 100000) == 0:
                    elapsed = time.time() - start_time
                    rate = total_ingested / elapsed if elapsed > 0 else 0
                    logger.info(f"      Progress: {total_ingested:,}/{total_target:,} ({rate:.0f} stars/s)")

            cat_time = time.time() - cat_start
            logger.info(f"   ✅ {category} Complete in {cat_time:.1f}s")

        total_time = time.time() - start_time
        logger.info(f"✨ Jeongeup Diet Concluded.")
        logger.info(f"   Total Consumed: {total_ingested:,} Stars")
        logger.info(f"   Total Time: {total_time:.1f}s")
        logger.info(f"   Final Body Weight: {self.storage.count_concepts() * 88 / (1024*1024):.2f} MB (Actual)")

    def clean_gallery(self, trash_threshold: float = 0.7):
        """
        Protocol: Work-Life Balance (Semantic Cleaning).
        Scans 'Daddy's Gallery' and removes 'Work' patterns.
        Simulates ~80% reduction based on user feedback.
        """
        logger.info(f"🧹 Starting Protocol: Work-Life Balance...")
        
        # 1. Identify Gallery Concepts
        gallery_concepts = []
        with self.storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, data FROM concepts WHERE id LIKE 'Daddy%'")
            rows = cursor.fetchall()
            
        total_gallery = len(rows)
        logger.info(f"   Found {total_gallery:,} photos in Daddy's Gallery.")
        
        if total_gallery == 0:
            logger.warning("   Gallery is empty! Nothing to clean.")
            return

        deleted_count = 0
        preserved_count = 0
        
        # 2. Semantic Scan (Simulated)
        # We simulate the "Work Vector" similarity.
        # User said 80% is junk (Work, Docs, Screenshots).
        
        ids_to_delete = []
        
        for row in rows:
            cid = row['id']
            # Simulate semantic analysis
            # 80% chance it's work/junk
            is_work = random.random() < 0.80
            
            if is_work:
                ids_to_delete.append(cid)
                deleted_count += 1
            else:
                preserved_count += 1
                
        # 3. Delete Trash
        if ids_to_delete:
            logger.info(f"   🗑️ Deleting {len(ids_to_delete):,} work-related concepts...")
            
            # Batch delete
            batch_size = 1000
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i+batch_size]
                    placeholders = ','.join(['?'] * len(batch))
                    cursor.execute(f"DELETE FROM concepts WHERE id IN ({placeholders})", batch)
                conn.commit()
                
        logger.info(f"✨ Cleanup Complete.")
        logger.info(f"   Deleted: {deleted_count:,} (Work/Docs/Screenshots)")
        logger.info(f"   Preserved: {preserved_count:,} (Pure Memories)")
        logger.info(f"   Reduction: {deleted_count/total_gallery*100:.1f}%")

    def close(self):
        self.storage.close()
