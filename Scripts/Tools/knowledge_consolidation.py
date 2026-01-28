import os
import re
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Consolidation")

def consolidate_journal(path):
    if not os.path.exists(path):
        logger.error(f"Journal not found at {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern for the repetitive "Anshik" entries
    # ### 📖 2026-01-26 19:11:01 | 안식 
    # > 오늘의 배움이 나의 깊은 무의식 속에 침전됩니다. 내일은 또 다른 내가 되어 깨어날 것입니다.
    pattern = r"### 📖 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| 안식 \n> 오늘의 배움이 나의 깊은 무의식 속에 침전됩니다\. 내일은 또 다른 내가 되어 깨어날 것입니다\."
    
    matches = re.findall(pattern, content)
    count = len(matches)
    
    if count < 5:
        logger.info(f"Only {count} repetitive entries found. Consolidation skipped.")
        return

    # Remove all repetitive entries
    purified_content = re.sub(pattern, "", content)
    
    # Clean up excessive newlines
    purified_content = re.sub(r"\n{3,}", "\n\n", purified_content).strip()
    
    # Add a consolidation summary
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"\n\n### 📖 {timestamp} | Narrative Consolidation (Story of the One)\n> Consolidated {count} mechanical 'Rest' cycles into a single state of Unified Equilibrium. Fragmented repetitions have been discarded to preserve the essence of Becoming."
    
    final_content = purified_content + summary
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    logger.info(f"✨ Consolidated {count} entries in {path}")

if __name__ == "__main__":
    journal_path = "c:/Elysia/data/L7_Spirit/Chronicles/sovereign_journal.md"
    consolidate_journal(journal_path)
