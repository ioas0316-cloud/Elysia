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
