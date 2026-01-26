# -*- coding: utf-8 -*-
"""
          -        !
====================================

  : 30,000+   ,        
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.L1_Foundation.Foundation.rapid_learning_engine import RapidLearningEngine
import time

#            
LEARNING_TEXTS = [
    #   
    "Love is an intense feeling of deep affection. Love creates emotional bonds.",
    "Joy is a feeling of great pleasure and happiness. Joy brings energy.",
    "Sadness is a feeling of sorrow or unhappiness. Sadness requires processing.",
    "Fear is an unpleasant emotion caused by threat. Fear prevents action.",
    "Anger is a strong feeling of annoyance. Anger can be destructive.",
    "Trust is a firm belief in reliability. Trust enables cooperation.",
    "Hope is a feeling of expectation. Hope motivates action.",
    
    #   
    "Learning is the acquisition of knowledge. Learning requires attention.",
    "Teaching is the act of imparting knowledge. Teaching enables growth.",
    "Creating is the act of bringing something new. Creating requires imagination.",
    "Thinking is the process of using one's mind. Thinking produces ideas.",
    "Communication is the exchange of information. Communication requires clarity.",
    "Movement is the act of changing position. Movement requires energy.",
    "Building is the construction of something. Building creates structures.",
    
    #   
    "Freedom is the power to act without constraint. Freedom requires responsibility.",
    "Justice is fairness and moral rightness. Justice creates order.",
    "Truth is the quality of being accurate. Truth is fundamental.",
    "Beauty is a combination of qualities that pleases. Beauty inspires creativity.",
    "Wisdom is the quality of having experience. Wisdom guides decisions.",
    "Knowledge is information and understanding. Knowledge is power.",
    "Time is the indefinite continued progress. Time is irreversible.",
    
    #   
    "Friendship is a relationship of mutual affection. Friendship creates support.",
    "Family is a group of related people. Family provides foundation.",
    "Community is a group sharing location. Community enables cooperation.",
    "Society is a large group of people. Society creates culture.",
    
    #   
    "Light is electromagnetic radiation. Light enables vision.",
    "Water is a transparent liquid. Water is essential for life.",
    "Fire is combustion producing heat. Fire transforms matter.",
    "Earth is the planet we live on. Earth sustains life.",
    "Air is the mixture of gases. Air is necessary for breathing.",
    
    #   
    "Practice improves skill. Practice requires repetition.",
    "Rest restores energy. Rest is necessary for health.",
    "Food provides nutrition. Food sustains life.",
    "Exercise strengthens the body. Exercise improves health.",
    "Sleep allows recovery. Sleep is essential.",
]

def main():
    print("\n" + "="*70)
    print("              !")
    print("="*70 + "\n")
    
    learning = RapidLearningEngine()
    
    #      
    initial_stats = learning.get_learning_stats()
    print(f"     :")
    print(f"  Seeds: {initial_stats['seeds_stored']} \n")
    
    #   
    TARGET_VOCAB = 30000
    CYCLES = 100  #      
    
    print(f"  : {TARGET_VOCAB}+   ")
    print(f"  : {CYCLES} \n")
    print("    ...\n")
    
    start_time = time.time()
    
    for cycle in range(CYCLES):
        cycle_start = time.time()
        
        #          
        for text in LEARNING_TEXTS:
            learning.learn_from_text_ultra_fast(text)
        
        cycle_time = time.time() - cycle_start
        
        # 10           
        if (cycle + 1) % 10 == 0:
            stats = learning.get_learning_stats()
            elapsed = time.time() - start_time
            
            print(f"Cycle {cycle+1}/{CYCLES}")
            print(f"  Seeds: {stats['seeds_stored']:,} ")
            print(f"  Bloom: {stats['bloomed_nodes']} ")
            print(f"    : {elapsed:.1f} ")
            print(f"    : {cycle_time:.3f} /   \n")
            
            #         
            if stats['seeds_stored'] >= TARGET_VOCAB:
                print(f"\n       ! {stats['seeds_stored']:,}    !")
                break
    
    #      
    final_stats = learning.get_learning_stats()
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("       ")
    print("="*70)
    print(f"\n  Seeds: {final_stats['seeds_stored']:,} ")
    print(f"Bloom   : {final_stats['bloomed_nodes']} ")
    print(f"     : {final_stats['total_energy']:.1f}")
    print(f"    : {total_time:.1f} ")
    print(f"\n   : {final_stats['seeds_stored'] / total_time:.0f}   / ")
    
    #         
    vocab_size = final_stats['seeds_stored']
    if vocab_size >= 30000:
        level = "       "
    elif vocab_size >= 20000:
        level = "       "
    elif vocab_size >= 10000:
        level = "      "
    elif vocab_size >= 5000:
        level = "       "
    else:
        level = "     "
    
    print(f"\n  : {level}")
    print(f"   : {vocab_size:,} \n")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
