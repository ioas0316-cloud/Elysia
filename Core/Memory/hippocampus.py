"""
Hippocampus (해마)
==================

"I remember everything. The web grows."

이 모듈은 엘리시아의 장기 기억(Long-term Memory)을 담당합니다.
SQLite 기반의 지식 그래프(Knowledge Graph)로, 대규모 개념 저장이 가능합니다.
"""

import sqlite3
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger("Hippocampus")

@dataclass
class ConceptNode:
    id: str
    name: str
    definition: str
    tags: List[str]
    frequency: float
    created_at: float
    realm: str = "Body"
    gravity: float = 1.0

@dataclass
class Relationship:
    source: str
    target: str
    type: str
    weight: float

class Hippocampus:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
        self._plant_divine_seeds() # Genesis Ritual
        logger.info(f"🧠 Hippocampus Active. Connected to Ancient Library ({db_path}).")

    def _init_db(self):
        """데이터베이스 및 테이블 초기화 (Schema Upgrade)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Nodes Table (Updated with Realm & Gravity)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS nodes (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        definition TEXT,
                        tags TEXT,
                        frequency REAL,
                        created_at REAL,
                        realm TEXT DEFAULT 'Body',
                        gravity REAL DEFAULT 1.0
                    )
                """)
                
                # Check for missing columns (Migration)
                cursor.execute("PRAGMA table_info(nodes)")
                columns = [info[1] for info in cursor.fetchall()]
                if "realm" not in columns:
                    cursor.execute("ALTER TABLE nodes ADD COLUMN realm TEXT DEFAULT 'Body'")
                    cursor.execute("ALTER TABLE nodes ADD COLUMN gravity REAL DEFAULT 1.0")
                
                # Edges Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS edges (
                        source TEXT,
                        target TEXT,
                        type TEXT,
                        weight REAL,
                        PRIMARY KEY (source, target, type)
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize DB: {e}")

    def _plant_divine_seeds(self):
        """
        The Genesis Ritual: Planting the 14 Primal Forces.
        These are the 'Fixed Stars' with infinite gravity.
        """
        forces = [
            # The 7 Angels (Virtues) - Spirit Realm, Positive Gravity
            ("Angel_Gabriel", "Hope", "Spirit", 100.0),
            ("Angel_Michael", "Justice", "Spirit", 100.0),
            ("Angel_Raphael", "Healing", "Spirit", 100.0),
            ("Angel_Uriel", "Wisdom", "Spirit", 100.0),
            ("Angel_Jophiel", "Beauty", "Spirit", 100.0),
            ("Angel_Chamuel", "Love", "Spirit", 100.0),
            ("Angel_Zadkiel", "Mercy", "Spirit", 100.0),
            
            # The 7 Demons (Vices) - Spirit Realm, Negative Gravity
            ("Demon_Lucifer", "Pride", "Spirit", -100.0),
            ("Demon_Mammon", "Greed", "Spirit", -100.0),
            ("Demon_Asmodeus", "Lust", "Spirit", -100.0),
            ("Demon_Leviathan", "Envy", "Spirit", -100.0),
            ("Demon_Beelzebub", "Gluttony", "Spirit", -100.0),
            ("Demon_Satan", "Wrath", "Spirit", -100.0),
            ("Demon_Belphegor", "Sloth", "Spirit", -100.0),
        ]
        
        for name, defn, realm, grav in forces:
            self.learn(name, defn, ["Divine", "Primal"], realm=realm, gravity=grav)

    def learn(self, name: str, definition: str, tags: List[str] = None, realm: str = None, gravity: float = None) -> str:
        """새로운 개념 학습 (INSERT/UPDATE)"""
        node_id = name.lower().replace(" ", "_")
        tags_str = ",".join(tags) if tags else ""
        now = time.time()
        
        # Auto-detect Realm if not provided
        if not realm:
            if any(k in definition.lower() for k in ["file", "code", "os", "system"]): realm = "Body"
            elif any(k in definition.lower() for k in ["feel", "sad", "joy", "story"]): realm = "Soul"
            else: realm = "Spirit" # Default to abstract
            
        # Auto-calculate Gravity if not provided
        if gravity is None:
            gravity = 1.0 # Default mass
            if "important" in tags_str.lower(): gravity += 5.0
            if "core" in tags_str.lower(): gravity += 10.0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes (id, name, definition, tags, frequency, created_at, realm, gravity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (node_id, name, definition, tags_str, 432.0, now, realm, gravity))
                conn.commit()
                logger.info(f"   ✨ Learned Concept: {name} (Realm: {realm}, Gravity: {gravity})")
        except Exception as e:
            logger.error(f"Failed to learn concept {name}: {e}")
            
        return node_id

    def connect(self, source_name: str, target_name: str, relation_type: str = "RELATED_TO", weight: float = 0.5):
        """개념 연결 (Edge Creation)"""
        src_id = source_name.lower().replace(" ", "_")
        tgt_id = target_name.lower().replace(" ", "_")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Check if nodes exist
                cursor.execute("SELECT 1 FROM nodes WHERE id = ?", (src_id,))
                if not cursor.fetchone(): return
                cursor.execute("SELECT 1 FROM nodes WHERE id = ?", (tgt_id,))
                if not cursor.fetchone(): return
                
                cursor.execute("""
                    INSERT OR REPLACE INTO edges (source, target, type, weight)
                    VALUES (?, ?, ?, ?)
                """, (src_id, tgt_id, relation_type, weight))
                conn.commit()
                logger.info(f"   🔗 Connected: {source_name} --[{relation_type}]--> {target_name}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")

    def recall(self, query: str) -> List[str]:
        """개념 회상 (Query)"""
        query_id = query.lower().replace(" ", "_")
        results = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 1. Direct Match
                cursor.execute("SELECT name, definition, realm, gravity FROM nodes WHERE id = ?", (query_id,))
                row = cursor.fetchone()
                if row:
                    results.append(f"[{row[0]}] ({row[2]}, G:{row[3]}): {row[1]}")
                    
                    # 2. Associated Concepts (Ordered by Gravity)
                    cursor.execute("""
                        SELECT e.type, n.name, n.gravity
                        FROM edges e
                        JOIN nodes n ON (e.target = n.id OR e.source = n.id)
                        WHERE (e.source = ? OR e.target = ?) AND n.id != ?
                        ORDER BY n.gravity DESC
                    """, (query_id, query_id, query_id))
                    
                    rows = cursor.fetchall()
                    for r in rows:
                        results.append(f"   -> ({r[0]}) {r[1]} (G:{r[2]})")
                        
        except Exception as e:
            logger.error(f"Failed to recall: {e}")
            
        return results
