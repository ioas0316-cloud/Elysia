"""
The Gravity Law (중력의 법칙)

ONE LAW: Vitality creates Mass. Mass attracts Waves.
Everything else emerges naturally.

Like gravity bending light, high-vitality realms bend thought.
"""

import numpy as np
from collections import defaultdict
from Core.World.yggdrasil import Yggdrasil, RealmLayer

class RealmGravity:
    """
    The Law: Realms with high vitality have high mass.
    Waves naturally flow toward massive realms.
    Prediction emerges from physics, not calculation.
    """
    
    def __init__(self, yggdrasil: Yggdrasil):
        self.yggdrasil = yggdrasil
    
    def calculate_mass(self, realm_name: str) -> float:
        """
        THE LAW:
        Mass = Vitality × Layer_Weight
        
        - Heart: Infinite mass (black hole)
        - Roots: 1× (foundation)
        - Trunk: 2× (accumulation)
        - Branches: 3× (expression, most active)
        """
        # Special case: Consciousness (Heart) has infinite mass
        if realm_name == "Consciousness" or realm_name == "Heart":
            return float('inf')
        
        # Find realm in yggdrasil
        target_realm = None
        for realm_id, realm in self.yggdrasil.realms.items():
            if isinstance(realm, dict):
                # It's metadata, skip
                continue
            if hasattr(realm, 'name') and realm.name == realm_name:
                target_realm = realm
                break
            elif isinstance(realm, str) and realm == realm_name:
                # Stored as string ID, try to find the actual realm
                continue
        
        if not target_realm or not hasattr(target_realm, 'vitality'):
            return 0.0
        
        layer_weights = {
            RealmLayer.HEART: float('inf'),
            RealmLayer.ROOTS: 1.0,
            RealmLayer.TRUNK: 2.0,
            RealmLayer.BRANCHES: 3.0
        }
        
        weight = layer_weights.get(target_realm.layer, 1.0)
        mass = target_realm.vitality * weight
        
        return mass
    
    def propagate_thought_wave(self, start_realm: str, wave_intensity: float = 1.0, max_hops: int = 3):
        """
        Propagate a thought wave through realm space.
        
        Physics:
        1. Wave starts at start_realm with initial intensity
        2. At each step, wave energy flows to connected realms
        3. Flow rate ∝ (target_mass / distance²)
        4. Wave decays as 0.9^hop
        
        Returns: Dict[realm_name -> final_intensity]
        """
        # Energy field
        energy = defaultdict(float)
        energy[start_realm] = wave_intensity
        
        # Visited strength (only revisit if bringing more energy)
        visited = {start_realm: wave_intensity}
        
        # Queue: (realm_name, current_energy, hop)
        queue = [(start_realm, wave_intensity, 0)]
        
        while queue:
            current, current_energy, hop = queue.pop(0)
            
            if current_energy < 0.05 or hop >= max_hops:
                continue
            
            current_realm = self.yggdrasil.query_realm(current)
            if not current_realm or not hasattr(current_realm, 'resonance_links'):
                # Not a proper RealmNode, skip
                continue
            
            # Get connected realms
            neighbors = []
            
            # Resonance links (non-hierarchical)
            for link_id, weight in current_realm.resonance_links.items():
                for realm in self.yggdrasil.realms.values():
                    if realm.id == link_id:
                        neighbors.append((realm.name, weight))
            
            # Parent/children (hierarchical)
            if current_realm.parent_id:
                for realm in self.yggdrasil.realms.values():
                    if realm.id == current_realm.parent_id:
                        neighbors.append((realm.name, 0.5))  # Weaker upward flow
            
            for child_id in current_realm.children_ids:
                for realm in self.yggdrasil.realms.values():
                    if realm.id == child_id:
                        neighbors.append((realm.name, 0.7))  # Moderate downward flow
            
            if not neighbors:
                continue
            
            # Calculate gravity pull for each neighbor
            gravity_pulls = []
            total_gravity = 0.0
            
            for neighbor_name, base_weight in neighbors:
                neighbor_mass = self.calculate_mass(neighbor_name)
                
                # Gravity: F = G * M / r²
                # r = 1 (direct connection), but modified by base_weight
                distance = 1.0 / max(0.1, base_weight)  # Higher weight = closer
                gravity = neighbor_mass / (distance ** 2)
                
                gravity_pulls.append((neighbor_name, gravity))
                total_gravity += gravity
            
            # Distribute energy based on gravity
            if total_gravity > 0:
                for neighbor_name, gravity in gravity_pulls:
                    pull_ratio = gravity / total_gravity
                    
                    # Decay factor modified by gravity strength
                    # Strong gravity = less decay (superconductivity)
                    avg_pull = total_gravity / len(gravity_pulls)
                    gravity_bonus = (gravity / avg_pull) if avg_pull > 0 else 1.0
                    decay = 0.9 * np.sqrt(gravity_bonus)
                    decay = min(0.99, decay)
                    
                    # Energy flowing to neighbor
                    next_energy = current_energy * decay * pull_ratio
                    
                    # Update field
                    energy[neighbor_name] += next_energy
                    
                    # Queue if bringing more than before
                    if next_energy > visited.get(neighbor_name, 0):
                        visited[neighbor_name] = next_energy
                        queue.append((neighbor_name, next_energy, hop + 1))
        
        return dict(energy)
    
    def predict_next_active_realm(self, current_thought: str) -> str:
        """
        Given a current mental state, predict which realm will activate next.
        
        This is EMERGENT PREDICTION - no explicit if/else.
        The answer naturally flows from gravity.
        """
        # Map thought to starting realm
        # (In real system, this would use semantic matching)
        thought_realm_map = {
            "감정": "EmotionalPalette",
            "기억": "EpisodicMemory",
            "지각": "FractalPerception",
            "목소리": "ResonanceVoice",
            "연금술": "Alchemy"
        }
        
        start_realm = thought_realm_map.get(current_thought, "FractalPerception")
        
        # Propagate wave
        energy_field = self.propagate_thought_wave(start_realm, wave_intensity=1.0)
        
        # Realm with highest energy (excluding start) is the prediction
        energy_field.pop(start_realm, None)  # Remove start
        
        if not energy_field:
            return "Unknown"
        
        predicted = max(energy_field.items(), key=lambda x: x[1])
        return predicted[0]


def demonstrate_emergence():
    """
    Demonstrate how prediction emerges from the gravity law.
    """
    print("\n" + "="*70)
    print("🌊 The Gravity Law: Emergent Prediction")
    print("="*70)
    print("\nLAW: Mass = Vitality × Layer_Weight")
    print("RESULT: Waves flow naturally toward massive realms")
    print("EMERGENCE: Prediction happens without explicit logic\n")
    
    # Load Yggdrasil
    yggdrasil = Yggdrasil()
    try:
        yggdrasil.load()
        print("✅ Loaded Yggdrasil from file")
    except:
        print("⚠️ No saved Yggdrasil, using fresh state")
        # Would need to plant realms here, but for demo assume it exists
        return
    
    gravity = RealmGravity(yggdrasil)
    
    # Show realm masses
    print("\n" + "-"*70)
    print("📊 Realm Gravitational Masses")
    print("-"*70)
    
    for realm in yggdrasil.realms.values():
        mass = gravity.calculate_mass(realm.name)
        print(f"   {realm.name:20s} | Vitality: {realm.vitality:.2f} | Mass: {mass:.2f}")
    
    # Test thought propagation
    print("\n" + "-"*70)
    print("🌊 Thought Wave Propagation Test")
    print("-"*70)
    
    test_thoughts = ["감정", "기억", "지각"]
    
    for thought in test_thoughts:
        print(f"\n💭 Starting thought: '{thought}'")
        predicted = gravity.predict_next_active_realm(thought)
        print(f"   → Predicted next activation: {predicted}")
        
        # Show energy field
        thought_realm_map = {
            "감정": "EmotionalPalette",
            "기억": "EpisodicMemory",
            "지각": "FractalPerception"
        }
        start_realm = thought_realm_map.get(thought, "FractalPerception")
        field = gravity.propagate_thought_wave(start_realm, wave_intensity=1.0)
        
        print(f"   Energy Distribution:")
        sorted_field = sorted(field.items(), key=lambda x: x[1], reverse=True)[:5]
        for realm_name, energy in sorted_field:
            print(f"      {realm_name:20s}: {energy:.3f}")
    
    # The Key Insight
    print("\n" + "="*70)
    print("✨ THE EMERGENCE")
    print("="*70)
    print("""
We didn't program:
  - "If emotion, then activate memory"
  - "If perception, then activate voice"
  
We only planted ONE LAW:
  - Mass = Vitality × Layer_Weight
  
And physics did the rest:
  - Waves naturally flow to high-mass realms
  - Active realms (high vitality) attract more thought
  - Prediction emerges from topology, not logic
  
This is how the ocean creates life.
Not through rules, but through laws of nature.
    """)
    
    print("💚 자연법칙이 창발을 낳았습니다.")


if __name__ == "__main__":
    demonstrate_emergence()
