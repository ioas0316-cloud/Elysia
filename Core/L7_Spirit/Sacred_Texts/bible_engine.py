import sqlite3
import json
import os
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Elysia.Spirit.Bible")

class BibleEngine:
    """
    The Living Word Engine.
    Handles high-speed Bible lookup, semantic search, and meditation synchronization.
    """
    def __init__(self, db_path: str = "data/L7_Spirit/Sacred_Texts/bible.db"):
        self.db_path = db_path
        self.cache = {}
        self._ensure_db()

    def _ensure_db(self):
        """Initializes the SQLite database if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for verses (Multiple translations support)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation TEXT,
                book TEXT,
                chapter INTEGER,
                verse INTEGER,
                content TEXT,
                UNIQUE(translation, book, chapter, verse)
            )
        ''')
        
        # Create FTS5 for ultra-fast search if available
        try:
            cursor.execute('CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(content, content="verses", content_rowid="id")')
        except sqlite3.OperationalError:
            logger.warning("FTS5 not supported in this environment. Falling back to simple LIKE search.")
            
        conn.commit()
        conn.close()

    def add_verse(self, translation: str, book: str, chapter: int, verse: int, content: str):
        """Seeds a single verse into the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO verses (translation, book, chapter, verse, content)
            VALUES (?, ?, ?, ?, ?)
        ''', (translation.upper(), book, chapter, verse, content))
        conn.commit()
        conn.close()

    def get_verse(self, book: str, chapter: int, verse: int, translation: str = "NIV") -> Optional[str]:
        """Retrieves a verse instantly."""
        cache_key = f"{translation}:{book}:{chapter}:{verse}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT content FROM verses 
            WHERE translation = ? AND book = ? AND chapter = ? AND verse = ?
        ''', (translation.upper(), book, chapter, verse))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.cache[cache_key] = result[0]
            return result[0]
        return None

    def search(self, query: str, translation: str = "NIV", limit: int = 10) -> List[Tuple[str, int, int, str]]:
        """Fast keyword search across the Bible."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Fallback to LIKE if FTS5 failed
        cursor.execute('''
            SELECT book, chapter, verse, content FROM verses 
            WHERE translation = ? AND content LIKE ? 
            LIMIT ?
        ''', (translation.upper(), f"%{query}%", limit))
        results = cursor.fetchall()
        conn.close()
        return results

    def seed_key_verses(self):
        """Initial seeding of foundational Bible Monads."""
        key_verses = [
            ("    ", 1, 1, "                                               ", "KR"),
            ("John", 1, 1, "In the beginning was the Word, and the Word was with God, and the Word was God.", "NIV"),
            ("    ", 7, 12, "                                                       ", "KR"),
            ("Matthew", 7, 12, "So in everything, do to others what you would have them do to you, for this sums up the Law and the Prophets.", "NIV"),
            ("    ", 8, 32, "                        ", "KR"),
            ("John", 8, 32, "Then you will know the truth, and the truth will set you free.", "NIV"),
        ]
        for book, ch, vs, content, trans in key_verses:
            self.add_verse(trans, book, ch, vs, content)
        logger.info(f"  [Sovereign Seed] {len(key_verses)} key verses seeded into L7 Memory.")

if __name__ == "__main__":
    engine = BibleEngine()
    engine.seed_key_verses()
    v = engine.get_verse("John", 1, 1)
    print(f"Loaded: {v}")