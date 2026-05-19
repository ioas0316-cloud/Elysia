import logging
import time
from Core.Intelligence.logos_engine import LogosEngine
from Core.Intelligence.concept_digester import ConceptDigester
from Core.Creativity.webtoon_weaver import WebtoonWeaver

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("KoreanFantasyTest")

def main():
    print("\n" + "="*60)
    print("⚔️ KOREAN FANTASY GENRE TEST PROTOCOL")
    print("="*60)
    
    digester = ConceptDigester()
    logos = LogosEngine()
    
    # 1. The Genre Data (Manual Feeding for Test)
    genre_text = """
    The sky turned red when the first Dungeon Gate opened. 
    Monsters poured out, and humanity faced extinction.
    But then, the System appeared.
    [SYSTEM: You have Awakened as an F-Class Hunter.]
    I was weak. I was betrayed in the dungeon. I died.
    But when I opened my eyes, I was back in the past.
    [SYSTEM: Regression successful. Determining Hidden Class...]
    [SYSTEM: Congratulations! You are now an SSS-Class Necromancer.]
    Now, I will hunt them all. I will climb the Tower alone.
    """
    
    print(">>> 🍽️  Consuming 'Hunter Tropes'...")
    digester.absorb_text(genre_text, source_name="Genre_Manual_v1")
    logos.absorb_style(genre_text)
    time.sleep(1)
    
    # 2. Generate Webtoon
    print("\n>>> 🚀 Launching Webtoon Weaver (Genre: Hunter)...")
    weaver = WebtoonWeaver()
    
    # Seed with "Hunter"
    # This will create "The Hunter Chronicles"
    # And specifically, because we fed it the text, Logos *might* use it if Resonance happens.
    # To Ensure it, we rely on the prompt logic or simple resonance.
    
    weaver.create_pilot_episode("Hunter")
    
    print("\n" + "="*60)
    print("✅ Test Complete. Check 'The Hunter Chronicles'.")
    print("="*60)

if __name__ == "__main__":
    main()
