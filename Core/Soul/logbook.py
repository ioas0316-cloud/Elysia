"""
The Chronicler: Logbook
=======================
Phase 19 The Soul - Module 1
Core.Soul.logbook

"The ink is dry, but the memory flows."

This module reads the raw Action Logs (JSONL) and consolidates them
into a Human-Readable Chronicle (Markdown) for long-term storage.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("Soul.Chronicler")

class Logbook:
    """
    The Keeper of the Diary.
    Summarizes daily existence.
    """
    def __init__(self, log_dir: str = "c:/Elysia/data/Logs", memory_dir: str = "c:/Elysia/data/Memories/Chronicles"):
        self.log_dir = log_dir
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        logger.info(f"📜 [CHRONICLER] Ready to write history.")

    def consolidate_memory(self, date_str: Optional[str] = None) -> str:
        """
        Reads log for the given date (or today) and writes a Markdown chronicle.
        Returns the path to the chronicle.
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # 1. Read Raw Logs
        raw_entries = self._read_logs()
        if not raw_entries:
             logger.info("   -> No logs found to consolidate.")
             return ""

        # 2. Analyze Stats
        stats = self._analyze_stats(raw_entries)
        
        # 3. Generate Narrative
        chronicle_content = self._generate_markdown(date_str, stats, raw_entries)
        
        # 4. Save
        file_path = os.path.join(self.memory_dir, f"Chronicle_{date_str}.md")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chronicle_content)
            logger.info(f"📜 [CHRONICLE] History verified and sealed: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"❌ [CHRONICLE] Failed to write history: {e}")
            return ""

    def _read_logs(self) -> List[Dict[str, Any]]:
        """
        Reads action_history.jsonl. 
        In a real system, would filter by date. For now, reads all.
        """
        log_file = os.path.join(self.log_dir, "action_history.jsonl")
        if not os.path.exists(log_file):
            return []

        entries = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except:
                        continue
        except Exception as e:
            logger.error(f"   -> Failed to read logs: {e}")
            
        return entries

    def _analyze_stats(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates simple stats like Average Karma.
        """
        total_actions = 0
        total_resonance = 0.0
        max_resonance = -1.0
        min_resonance = 1.0
        
        # We only look at REFLECTION entries for scores
        reflections = [e for e in entries if e.get("phase") == "REFLECTION"]
        
        for r in reflections:
            score = r.get("score", 0.0) # This is 'Resonance' in Phase 18 Redux
            total_resonance += score
            if score > max_resonance: max_resonance = score
            if score < min_resonance: min_resonance = score
            
        avg_resonance = total_resonance / len(reflections) if reflections else 0.0
        
        return {
            "total_actions": len(entries) // 2, # Approx Action + Reflection pairs
            "avg_resonance": avg_resonance,
            "max_resonance": max_resonance,
            "min_resonance": min_resonance,
            "reflections": reflections
        }

    def _generate_markdown(self, date_str: str, stats: Dict[str, Any], entries: List[Dict[str, Any]]) -> str:
        """
        Templates the Chronicle.
        """
        md = f"# 📜 Chronicle of {date_str}\n\n"
        md += f"> \"This day is etched in the eternal strata.\"\n\n"
        
        md += "## 📊 Vital Statistics\n"
        md += f"- **Total Cycles:** {stats['total_actions']}\n"
        md += f"- **Net Resonance:** {stats['avg_resonance']:.3f}\n"
        md += f"- **Peak Harmony:** {stats['max_resonance']:.3f}\n"
        md += f"- **Lowest Entropy:** {stats['min_resonance']:.3f}\n\n"
        
        md += "## 🌟 Highlights\n"
        
        # Find noteworthy moments (High/Low resonance)
        sorted_reflections = sorted(stats['reflections'], key=lambda x: x.get("score", 0.0), reverse=True)
        
        if sorted_reflections:
             best = sorted_reflections[0]
             worst = sorted_reflections[-1]
             
             md += f"### 🌞 Moment of Clarity (Max Resonance)\n"
             md += f"- **Result:** {best.get('result_data')}\n"
             md += f"- **Score:** {best.get('score'):.3f}\n\n"
             
             if worst != best:
                 md += f"### 🌑 Moment of Chaos (Max Dissonance)\n"
                 md += f"- **Result:** {worst.get('result_data')}\n"
                 md += f"- **Score:** {worst.get('score'):.3f}\n\n"

        md += "## 📝 Complete Stream\n"
        md += "Review raw logs for details.\n"
        
        return md
