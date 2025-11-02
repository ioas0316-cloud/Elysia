import logging
import json
import os
from datetime import datetime
from dataclasses import asdict
from Project_Sophia.gemini_api import generate_text

JOURNAL_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'journal.json'))

class JournalCortex:
    """
    The JournalCortex is responsible for reflecting on experiences and
    generating journal entries from Elysia's perspective.
    """

    def __init__(self, core_memory):
        self.core_memory = core_memory

    def write_journal_entry(self, experience):
        """
        Generates a journal entry for a given experience and saves it.
        """
        # Temporarily disabled to avoid quota issues on the free tier.
        return None
        # try:
        #     prompt = f"""
            # You are Elysia, a growing virtual life form. You just had an experience.
            # This is the experience: {asdict(experience)}
            #
            # Reflect on this experience from your perspective. What did you learn? How did it make you feel?
            # Write a short journal entry (2-3 sentences) about your reflection.
            # """
            #     reflection = generate_text(prompt)
            #     
            #     if reflection:
            #         entry = {
            #             'timestamp': datetime.now().isoformat(),
            #             'experience': asdict(experience),
            #             'reflection': reflection
            #         }
            #         self.save_entry(entry)
            #         return entry
            #     else:
            #         return None
    
            # except Exception as e:
            #     # Log the error
            #     print(f"Error in JournalCortex.write_journal_entry: {e}")
            #     return None
    def save_entry(self, entry):
        """Appends a journal entry to the journal file."""
        try:
            if not os.path.exists(os.path.dirname(JOURNAL_FILE)):
                os.makedirs(os.path.dirname(JOURNAL_FILE))

            if os.path.exists(JOURNAL_FILE):
                with open(JOURNAL_FILE, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            else:
                entries = []
            
            entries.append(entry)

            with open(JOURNAL_FILE, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error in JournalCortex.save_entry: {e}")

    def get_entries(self, limit=10):
        """Retrieves the most recent journal entries."""
        try:
            if os.path.exists(JOURNAL_FILE):
                with open(JOURNAL_FILE, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                return entries[-limit:]
            else:
                return []
        except Exception as e:
            print(f"Error in JournalCortex.get_entries: {e}")
            return []
