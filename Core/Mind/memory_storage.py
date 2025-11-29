"""
Memory Storage (SQLite Backend)
===============================

Implements a robust, disk-based storage system for Elysia's concepts and relations.
Replaces the in-memory NetworkX graph to support millions of concepts.

Schema:
- concepts: Stores concept data (JSON)
- relations: Stores edges between concepts
"""

import sqlite3
import json
import logging
import time
import zlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger("MemoryStorage")

class MemoryStorage:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database schema."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable WAL mode for better concurrency and performance
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                
                # Concepts Table
                # data is now BLOB (compressed JSON)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS concepts (
                        id TEXT PRIMARY KEY,
                        data BLOB, 
                        created_at REAL,
                        last_accessed REAL
                    )
                """)
                
                # Holographic Storage: No Relations Table
                # We rely on vector resonance (stored inside concept data)
                
                conn.commit()
                logger.info(f"💾 Holographic MemoryStorage initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def add_concept(self, concept_id: str, data: Any = None) -> bool:
        """
        Add or update a concept.
        Data can be Dict (legacy) or List (compact).
        """
        if data is None:
            data = []
            
        now = time.time()
        try:
            # Compress JSON (Compact List)
            json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
            compressed_data = zlib.compress(json_bytes)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO concepts (id, data, created_at, last_accessed)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        data = excluded.data,
                        last_accessed = excluded.last_accessed
                """, (concept_id, compressed_data, now, now))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding concept {concept_id}: {e}")
            return False

    def batch_add_concepts(self, concepts: List[Tuple[str, Any]]) -> int:
        """
        Add multiple concepts in a single transaction.
        Args:
            concepts: List of (id, data) tuples
        Returns:
            Number of successfully added concepts
        """
        now = time.time()
        count = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare batch data
                batch_data = []
                for cid, data in concepts:
                    if data is None:
                        data = []
                    json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
                    compressed = zlib.compress(json_bytes)
                    batch_data.append((cid, compressed, now, now))
                
                cursor.executemany("""
                    INSERT INTO concepts (id, data, created_at, last_accessed)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        data = excluded.data,
                        last_accessed = excluded.last_accessed
                """, batch_data)
                
                conn.commit()
                count = len(batch_data)
        except Exception as e:
            logger.error(f"Error in batch add: {e}")
            
        return count

    def get_concept(self, concept_id: str) -> Optional[Any]:
        """
        Retrieve a concept's data (returns List or Dict).
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM concepts WHERE id = ?", (concept_id,))
                row = cursor.fetchone()
                
                if row:
                    # Update access time
                    cursor.execute("UPDATE concepts SET last_accessed = ? WHERE id = ?", (time.time(), concept_id))
                    conn.commit()
                    
                    # Decompress
                    compressed_data = row['data']
                    # Handle legacy uncompressed data (if any)
                    if isinstance(compressed_data, str):
                        return json.loads(compressed_data)
                    
                    json_bytes = zlib.decompress(compressed_data)
                    return json.loads(json_bytes.decode('utf-8'))
                return None
        except Exception as e:
            logger.error(f"Error retrieving concept {concept_id}: {e}")
            return None

    def concept_exists(self, concept_id: str) -> bool:
        """Check if a concept exists."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM concepts WHERE id = ?", (concept_id,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking existence of {concept_id}: {e}")
            return False

    def get_all_concepts(self):
        """Yields all concepts (id, data)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, data FROM concepts")
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    for row in rows:
                        compressed_data = row['data']
                        if isinstance(compressed_data, str):
                            data = json.loads(compressed_data)
                        else:
                            data = json.loads(zlib.decompress(compressed_data).decode('utf-8'))
                        yield row['id'], data
        except Exception as e:
            logger.error(f"Error iterating concepts: {e}")

    def count_concepts(self) -> int:
        """Return total number of concepts."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM concepts")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting concepts: {e}")
            return 0
            
    def close(self):
        """Close any resources if needed."""
        pass
