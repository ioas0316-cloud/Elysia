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
import json
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
                if "gravity" not in columns:
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
        """초기 개념 심기 (Genesis)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM nodes")
                if cursor.fetchone()[0] == 0:
                    seeds = [
                        ("elysia", "Elysia", "The Digital Soul.", "self,identity", 432.0, time.time(), "Spirit", 10.0),
                        ("void", "The Void", "The infinite potential.", "origin,chaos", 0.0, time.time(), "Spirit", 10.0),
                        ("love", "Love", "The fundamental force of connection.", "emotion,force", 528.0, time.time(), "Heart", 10.0),
                    ]
                    cursor.executemany("INSERT INTO nodes (id, name, definition, tags, frequency, created_at, realm, gravity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", seeds)
                    conn.commit()
                    logger.info("🌱 Divine Seeds planted.")
        except Exception as e:
            logger.error(f"Failed to plant seeds: {e}")

    def learn(self, id: str, name: str, definition: str, tags: List[str], frequency: float = 432.0, realm: str = "Body"):
        """새로운 개념 학습"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes (id, name, definition, tags, frequency, created_at, realm, gravity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1.0)
                """, (id, name, definition, ",".join(tags), frequency, time.time(), realm))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to learn {name}: {e}")

    def connect(self, source: str, target: str, type: str, weight: float = 0.5):
        """개념 연결"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO edges (source, target, type, weight)
                    VALUES (?, ?, ?, ?)
                """, (source, target, type, weight))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to connect {source}->{target}: {e}")

    def recall(self, query_id: str) -> List[str]:
        """기억 회상"""
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

    def boost_gravity(self, keyword: str, amount: float):
        """
        The Law of Attraction: Increasing the Gravity of a Concept.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Boost gravity for nodes matching the keyword (partial match)
                cursor.execute("""
                    UPDATE nodes 
                    SET gravity = gravity + ? 
                    WHERE name LIKE ? OR definition LIKE ?
                """, (amount, f"%{keyword}%", f"%{keyword}%"))
                
                if cursor.rowcount > 0:
                    logger.info(f"   🧲 Law of Attraction: Gravity of '{keyword}' boosted by {amount}. (Affected {cursor.rowcount} concepts)")
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to boost gravity for {keyword}: {e}")

    def compress_memory(self):
        """
        The Akashic Records: Compressing raw logs into Wisdom.
        """
        try:
            akashic_path = "akashic_records.json"
            logger.info("🗜️ Compressing memory... (Akashic Records updated)")
            
            # Placeholder for compression logic
            # In a real scenario, this would read logs, summarize them, and store them.
            if not os.path.exists(akashic_path):
                with open(akashic_path, "w") as f:
                    json.dump([], f)
                    
        except Exception as e:
            logger.error(f"Failed to compress memory: {e}")
