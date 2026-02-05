
import sys
import os
import time

sys.path.append(os.getcwd())

from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S1_Body.L5_Mental.Reasoning.web_walker import WebWalker
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger

def verify_real_hand():
    logger = SomaticLogger("VERIFY_REALITY")
    logger.action("Initiating Reality Connection Test...")
    
    walker = WebWalker()
    
    # query = "Current year and global population 2026"
    # We ask something specific to check if it's not a hallucination
    query = "latest advancements in quantum computing 2025"
    
    logger.thought(f"Querying the World: '{query}'")
    
    try:
        result = walker.search(query)
        
        if result and result.get('results'):
             logger.action("Received signal from the Outside World.")
             for item in result['results']:
                 logger.action(f"  - [{item['rank']}] {item['title']} ({item['url']})")
                 if "internal://" in item['url']:
                     logger.admonition("⚠️ WARNING: This is a HALLUCINATION (Simulation Fallback).")
                 
             logger.thought("The Hand is functional. The Void is populated.")
        else:
             logger.admonition("The Hand grasped nothing. (No results returned)")
             
    except Exception as e:
        logger.admonition(f"Reality Connection Failed: {e}")

if __name__ == "__main__":
    verify_real_hand()
