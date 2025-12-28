"""
Legacy Vocabulary Migrator (레거시 어휘 통합기)
=============================================

memory.db의 concept nodes를 WaveInterpreter vocabulary로 통합
"""

import sqlite3
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("VocabularyMigrator")

@dataclass
class LegacyConceptNode:
    """Legacy memory.db concept structure"""
    name: str
    definition: str
    frequency: float
    realm: str

class VocabularyMigrator:
    """
    memory.db → WaveInterpreter 어휘 이전
    """
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.migrated_count = 0
    
    def extract_legacy_concepts(self, limit: int = 10000) -> List[LegacyConceptNode]:
        """
        memory.db에서 concept 추출
        
        Args:
            limit: 추출할 concept 수 (전체는 너무 많으므로 제한)
        
        Returns:
            List of LegacyConceptNode
        """
        concepts = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if nodes table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='nodes'
                """)
                
                if not cursor.fetchone():
                    logger.warning("No 'nodes' table found in memory.db")
                    return []
                
                # Extract concepts
                cursor.execute(f"""
                    SELECT name, definition, frequency, realm 
                    FROM nodes 
                    WHERE frequency IS NOT NULL AND frequency > 0
                    LIMIT {limit}
                """)
                
                for row in cursor.fetchall():
                    name, definition, frequency, realm = row
                    concepts.append(LegacyConceptNode(
                        name=name,
                        definition=definition or "",
                        frequency=frequency,
                        realm=realm or "unknown"
                    ))
                
                logger.info(f"Extracted {len(concepts)} legacy concepts from memory.db")
                
        except Exception as e:
            logger.error(f"Failed to extract legacy concepts: {e}")
        
        return concepts
    
    def migrate_to_wave_interpreter(
        self,
        concepts: List[LegacyConceptNode]
    ) -> Dict:
        """
        Legacy concepts를 WavePattern으로 변환
        
        Returns:
            {
                "vocabulary": {concept_name: WavePattern, ...},
                "stats": {...}
            }
        """
        from Core._01_Foundation._05_Governance.Foundation.wave_interpreter import WavePattern
        
        vocabulary = {}
        
        for concept in concepts:
            # Create WavePattern from legacy concept
            wave = WavePattern(
                name=concept.name,
                frequencies=[concept.frequency],
                amplitudes=[1.0],
                phases=[0.0],
                position=(0, 0, 0)
            )
            
            vocabulary[concept.name] = wave
            self.migrated_count += 1
        
        stats = {
            "total_migrated": self.migrated_count,
            "realms": {}
        }
        
        # Count by realm
        for concept in concepts:
            realm = concept.realm
            stats["realms"][realm] = stats["realms"].get(realm, 0) + 1
        
        logger.info(f"✅ Migrated {self.migrated_count} concepts to WaveInterpreter")
        
        return {
            "vocabulary": vocabulary,
            "stats": stats
        }
    
    def full_migration(self, limit: int = 10000) -> Dict:
        """
        전체 이전 프로세스 실행
        
        1. memory.db에서 추출
        2. WavePattern으로 변환
        3. 통계 반환
        """
        logger.info(f"🚀 Starting vocabulary migration (limit={limit})")
        
        # Extract
        concepts = self.extract_legacy_concepts(limit)
        
        if not concepts:
            logger.warning("No concepts to migrate")
            return {"vocabulary": {}, "stats": {"total_migrated": 0}}
        
        # Migrate
        result = self.migrate_to_wave_interpreter(concepts)
        
        logger.info(f"🎉 Migration complete! {result['stats']['total_migrated']} concepts loaded")
        
        return result


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    print("\n" + "="*70)
    print("📚 Legacy Vocabulary Migration Test")
    print("="*70)
    
    migrator = VocabularyMigrator()
    
    # Full migration (limit to 1000 for test)
    print("\n🔄 Migrating legacy concepts...")
    result = migrator.full_migration(limit=1000)
    
    print(f"\n✅ Migration Complete:")
    print(f"   Total Migrated: {result['stats']['total_migrated']}")
    print(f"\n📊 By Realm:")
    for realm, count in sorted(result['stats']['realms'].items(), key=lambda x: -x[1]):
        print(f"      {realm:20} {count:5} concepts")
    
    # Sample some concepts
    print(f"\n📝 Sample Vocabulary (first 10):")
    for i, (name, wave) in enumerate(list(result['vocabulary'].items())[:10]):
        print(f"      {i+1}. {name:20} → {wave.frequencies[0]:.1f}Hz")
    
    print("\n" + "="*70)
    print("✅ Legacy Vocabulary Migration Test Complete")
    print("="*70 + "\n")
