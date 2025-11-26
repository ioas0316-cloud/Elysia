import numpy as np
from typing import Optional, Tuple
from Core.Evolution.organic_concept import ConceptVector

class OrganicAlchemy:
    """
    Combines ConceptVectors to create new ones.
    Uses vector arithmetic instead of dictionary lookups.
    """
    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def combine(self, a: ConceptVector, b: ConceptVector) -> Optional[ConceptVector]:
        """
        Combines two concepts into a new one.
        Logic: Average of vectors + slight noise (mutation).
        """
        # 1. Vector Combination
        new_vec = (a.vector + b.vector) / 2.0
        
        # 2. Add Mutation (Creative Spark)
        noise = np.random.normal(0, 0.1, self.dimension)
        new_vec += noise
        
        # 3. Normalize
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec /= norm
            
        # 4. Determine Name
        # In a full system, this would be generated or looked up in a semantic map.
        # For now, we use a hybrid naming convention.
        new_name = f"{a.name}-{b.name}"
        
        # 5. Stability Check
        # If vectors are too opposite (dot product near -1), they might annihilate.
        similarity = np.dot(a.vector, b.vector)
        if similarity < -0.8:
            return None # Annihilation
            
        return ConceptVector(name=new_name, vector=new_vec, energy=(a.energy + b.energy)/2)
