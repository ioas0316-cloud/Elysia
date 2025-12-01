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
        logger.info(f"🧠 Hippocampus Active. Connected to Ancient Library ({db_path}).")

    def _init_db(self):
        """데이터베이스 및 테이블 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Nodes Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS nodes (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        definition TEXT,
                        tags TEXT,
                        frequency REAL,
                        created_at REAL
                    )
                """)
                
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

    def learn(self, name: str, definition: str, tags: List[str] = None) -> str:
        """새로운 개념 학습 (INSERT/UPDATE)"""
        node_id = name.lower().replace(" ", "_")
        tags_str = ",".join(tags) if tags else ""
        now = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes (id, name, definition, tags, frequency, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (node_id, name, definition, tags_str, 432.0, now))
                conn.commit()
                logger.info(f"   ✨ Learned/Updated Concept: {name}")
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
                cursor.execute("SELECT name, definition FROM nodes WHERE id = ?", (query_id,))
                row = cursor.fetchone()
                if row:
                    results.append(f"[{row[0]}]: {row[1]}")
                    
                    # 2. Associated Concepts
                    cursor.execute("""
                        SELECT e.type, n.name 
                        FROM edges e
                        JOIN nodes n ON (e.target = n.id OR e.source = n.id)
                        WHERE (e.source = ? OR e.target = ?) AND n.id != ?
                    """, (query_id, query_id, query_id))
                    
                    rows = cursor.fetchall()
                    for r in rows:
                        results.append(f"   -> ({r[0]}) {r[1]}")
                        
        except Exception as e:
            logger.error(f"Failed to recall: {e}")
            
        return results
