# -*- coding: utf-8 -*-
"""
        
==============

                 
"""

#         
KOREAN_MAPPING = {
    #   
    'Love': '  ',
    'Joy': '  ',
    'Sadness': '  ',
    'Fear': '   ',
    'Anger': '  ',
    'Trust': '  ',
    'Hope': '  ',
    'Happiness': '  ',
    
    #   
    'Learning': '  ',
    'Teaching': '   ',
    'Creating': '  ',
    'Thinking': '  ',
    'Communication': '  ',
    'Movement': '   ',
    'Building': '  ',
    
    #   
    'Freedom': '  ',
    'Justice': '  ',
    'Truth': '  ',
    'Beauty': '    ',
    'Wisdom': '  ',
    'Knowledge': '  ',
    'Time': '  ',
    
    #   
    'Friendship': '  ',
    'Family': '  ',
    'Community': '   ',
    'Society': '  ',
    
    #   
    'Light': ' ',
    'Water': ' ',
    'Fire': ' ',
    'Earth': ' ',
    'Air': '  ',
    'Ocean': '  ',
    'Forest': ' ',
    'Mountain': ' ',
    'River': ' ',
    'Desert': '  ',
    'Climate': '  ',
    'Season': '  ',
    
    #   
    'Physics': '   ',
    'Chemistry': '  ',
    'Biology': '   ',
    'Astronomy': '   ',
    'Geology': '   ',
    'Mathematics': '  ',
    'Ecology': '   ',
    'Genetics': '   ',
    
    #   
    'Computer': '   ',
    'Internet': '   ',
    'Software': '     ',
    'Algorithm': '    ',
    'Data': '   ',
    'Programming': '     ',
    
    #   
    'Music': '  ',
    'Painting': '  ',
    'Sculpture': '  ',
    'Dance': ' ',
    'Poetry': ' ',
    'Theater': '  ',
    'Architecture': '  ',
    
    #   
    'Politics': '  ',
    'Economics': '  ',
    'Law': ' ',
    'Education': '  ',
    'Culture': '  ',
    'Religion': '  ',
    'Ethics': '  ',
    
    #   
    'Consciousness': '  ',
    'Memory': '  ',
    'Perception': '  ',
    'Reasoning': '  ',
    'Creativity': '   ',
    'Will': '  ',
    'Identity': '   ',
    
    #      
    'Cooperation': '  ',
    'Competition': '  ',
    'Conflict': '  ',
    'Collaboration': '  ',
    'Leadership': '   ',
    'Empathy': '  ',
    
    #   
    'Infinity': '  ',
    'Eternity': '  ',
    'Nothing': ' ',
    'Everything': '  ',
    'Possible': '  ',
    'Necessary': '  ',
    'Contingent': '  ',
}

def get_korean_name(english_name: str) -> str:
    """            """
    return KOREAN_MAPPING.get(english_name, "")

def add_mapping(english: str, korean: str):
    """        (       )"""
    KOREAN_MAPPING[english] = korean
