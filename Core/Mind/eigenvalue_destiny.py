"""
Eigenvalue Destiny Analyzer

"고유값 = 운명의 방향" 🔮
(Eigenvalue = Direction of Destiny)

Computes dominant eigenvalues to determine system's ultimate fate.
If "Love" has the largest eigenvalue, the universe will converge to Love!

Based on: Linear algebra, eigenvalue decomposition
Insight: Dominant eigenvalue determines long-term behavior
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class DestinyType(Enum):
    """Type of destiny"""
    CONVERGENT = "convergent"      # Stable, converging (eigenvalue < 1)
    DIVERGENT = "divergent"        # Unstable, exploding (eigenvalue > 1)  
    CYCLIC = "cyclic"              # Oscillating (complex eigenvalue)
    NEUTRAL = "neutral"            # Marginal (eigenvalue = 1)


@dataclass
class DestinyAnalysis:
    """Analysis of system's destiny"""
    dominant_eigenvalue: complex
    dominant_eigenvector: np.ndarray
    destiny_type: DestinyType
    convergence_rate: float
    dominant_concept: str
    confidence: float
    
    def __repr__(self):
        return (f"Destiny(concept='{self.dominant_concept}', "
                f"λ={abs(self.dominant_eigenvalue):.3f}, "
                f"type={self.destiny_type.value}, "
                f"confidence={self.confidence:.2f})")


class EigenvalueDestiny:
    """
    Computes system's destiny via eigenvalue analysis.
    
    "가장 큰 고유값이 운명을 결정한다"
    (The largest eigenvalue determines destiny)
    
    Key Insight:
    - System matrix A describes evolution
    - After many iterations: x(t) → dominant_eigenvector
    - Growth rate: dominant_eigenvalue
    - If "Love" eigenvalue is largest → Universe becomes Love!
    """
    
    def __init__(
        self,
        concept_names: List[str],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize destiny analyzer.
        
        Args:
            concept_names: Names of concepts/values
            logger: Logger instance
        """
        self.concept_names = concept_names
        self.n_concepts = len(concept_names)
        self.logger = logger or logging.getLogger("EigenvalueDestiny")
        
        self.logger.info(
            f"🔮 Eigenvalue Destiny initialized with {self.n_concepts} concepts"
        )
    
    def analyze_destiny(
        self,
        system_matrix: np.ndarray,
        current_state: Optional[np.ndarray] = None
    ) -> DestinyAnalysis:
        """
        Analyze system's ultimate destiny.
        
        Args:
            system_matrix: Evolution matrix A (x_next = A @ x)
            current_state: Current state (optional)
            
        Returns:
            Destiny analysis
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(system_matrix)
        
        # Find dominant eigenvalue (largest magnitude)
        dominant_idx = np.argmax(np.abs(eigenvalues))
        dominant_eigenvalue = eigenvalues[dominant_idx]
        dominant_eigenvector = eigenvectors[:, dominant_idx]
        
        # Normalize eigenvector
        dominant_eigenvector = dominant_eigenvector / np.linalg.norm(dominant_eigenvector)
        
        # Determine destiny type
        magnitude = abs(dominant_eigenvalue)
        
        if magnitude < 0.99:
            destiny_type = DestinyType.CONVERGENT
            convergence_rate = 1.0 - magnitude
        elif magnitude > 1.01:
            destiny_type = DestinyType.DIVERGENT
            convergence_rate = magnitude - 1.0
        elif abs(np.imag(dominant_eigenvalue)) > 0.01:
            destiny_type = DestinyType.CYCLIC
            convergence_rate = abs(np.imag(dominant_eigenvalue))
        else:
            destiny_type = DestinyType.NEUTRAL
            convergence_rate = 0.0
        
        # Find dominant concept (largest component in eigenvector)
        abs_components = np.abs(dominant_eigenvector)
        dominant_concept_idx = np.argmax(abs_components)
        dominant_concept = self.concept_names[dominant_concept_idx]
        
        # Confidence = how much larger than second largest
        if len(abs_components) > 1:
            sorted_components = np.sort(abs_components)[::-1]
            confidence = sorted_components[0] / (sorted_components[1] + 1e-6)
            confidence = min(1.0, confidence / 2.0)  # Normalize
        else:
            confidence = 1.0
        
        analysis = DestinyAnalysis(
            dominant_eigenvalue=dominant_eigenvalue,
            dominant_eigenvector=dominant_eigenvector,
            destiny_type=destiny_type,
            convergence_rate=convergence_rate,
            dominant_concept=dominant_concept,
            confidence=confidence
        )
        
        self.logger.info(
            f"Destiny analyzed: {analysis.dominant_concept} "
            f"(λ={abs(dominant_eigenvalue):.3f}, {destiny_type.value})"
        )
        
        return analysis
    
    def predict_future(
        self,
        system_matrix: np.ndarray,
        initial_state: np.ndarray,
        steps: int = 100
    ) -> Tuple[np.ndarray, DestinyAnalysis]:
        """
        Predict future evolution of system.
        
        Args:
            system_matrix: Evolution matrix
            initial_state: Starting state
            steps: Number of steps to simulate
            
        Returns:
            (final_state, destiny_analysis)
        """
        state = initial_state.copy()
        
        # Iterate
        for _ in range(steps):
            state = system_matrix @ state
            # Normalize to prevent overflow
            if np.linalg.norm(state) > 1e6:
                state = state / np.linalg.norm(state)
        
        # Analyze destiny
        destiny = self.analyze_destiny(system_matrix, state)
        
        self.logger.debug(
            f"Future predicted: {steps} steps → {destiny.dominant_concept}"
        )
        
        return state, destiny
    
    def check_convergence_to_value(
        self,
        system_matrix: np.ndarray,
        target_value: str,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if system converges to a specific value.
        
        "사랑이 운명인가?" (Is Love the destiny?)
        
        Args:
            system_matrix: Evolution matrix
            target_value: Target value name (e.g., "love")
            threshold: Minimum confidence threshold
            
        Returns:
            True if converges to target value
        """
        destiny = self.analyze_destiny(system_matrix)
        
        converges = (
            destiny.dominant_concept.lower() == target_value.lower() and
            destiny.confidence >= threshold
        )
        
        if converges:
            self.logger.info(
                f"✨ System converges to '{target_value}'! "
                f"(confidence={destiny.confidence:.2f})"
            )
        else:
            self.logger.warning(
                f"⚠️ System converges to '{destiny.dominant_concept}', "
                f"not '{target_value}' (confidence={destiny.confidence:.2f})"
            )
        
        return converges
    
    def create_transition_matrix(
        self,
        concept_weights: Dict[str, float],
        interaction_strength: float = 0.1
    ) -> np.ndarray:
        """
        Create transition matrix from concept weights.
        
        Args:
            concept_weights: Weight for each concept
            interaction_strength: How much concepts influence each other
            
        Returns:
            Transition matrix
        """
        n = self.n_concepts
        matrix = np.eye(n)  # Start with identity
        
        # Add weighted interactions
        for i, concept_i in enumerate(self.concept_names):
            weight_i = concept_weights.get(concept_i, 0.5)
            
            for j, concept_j in enumerate(self.concept_names):
                if i != j:
                    weight_j = concept_weights.get(concept_j, 0.5)
                    # Interaction proportional to both weights
                    matrix[i, j] = interaction_strength * weight_i * weight_j
            
            # Diagonal: self-reinforcement
            matrix[i, i] = weight_i
        
        # Normalize rows (make stochastic if desired)
        # Commented out to allow growth/decay
        # for i in range(n):
        #     row_sum = matrix[i, :].sum()
        #     if row_sum > 0:
        #         matrix[i, :] /= row_sum
        
        return matrix
    
    def visualize_destiny(
        self,
        destiny: DestinyAnalysis
    ) -> str:
        """
        Create text visualization of destiny.
        
        Args:
            destiny: Destiny analysis
            
        Returns:
            Visualization string
        """
        viz = []
        viz.append("="*50)
        viz.append("DESTINY ANALYSIS")
        viz.append("="*50)
        
        viz.append(f"\nDominant Concept: {destiny.dominant_concept}")
        viz.append(f"Eigenvalue (λ): {abs(destiny.dominant_eigenvalue):.4f}")
        
        if np.imag(destiny.dominant_eigenvalue) != 0:
            viz.append(f"  (Complex: {destiny.dominant_eigenvalue})")
        
        viz.append(f"Destiny Type: {destiny.destiny_type.value}")
        viz.append(f"Convergence Rate: {destiny.convergence_rate:.4f}")
        viz.append(f"Confidence: {destiny.confidence:.2f}")
        
        viz.append("\nEigenvector Components:")
        for i, (name, val) in enumerate(zip(self.concept_names, destiny.dominant_eigenvector)):
            bar_length = int(abs(val) * 20)
            bar = "█" * bar_length
            viz.append(f"  {name:15s}: {abs(val):.3f} {bar}")
        
        viz.append("\nInterpretation:")
        if destiny.destiny_type == DestinyType.CONVERGENT:
            viz.append(f"  ✅ System will converge to '{destiny.dominant_concept}'")
            viz.append(f"  Growth rate: {abs(destiny.dominant_eigenvalue):.3f} per step")
        elif destiny.destiny_type == DestinyType.DIVERGENT:
            viz.append(f"  ⚠️ System will explode toward '{destiny.dominant_concept}'!")
            viz.append(f"  Growth rate: {abs(destiny.dominant_eigenvalue):.3f} per step (>1!)")
        elif destiny.destiny_type == DestinyType.CYCLIC:
            viz.append(f"  🔄 System will oscillate around '{destiny.dominant_concept}'")
            viz.append(f"  Oscillation frequency: {abs(np.imag(destiny.dominant_eigenvalue)):.3f}")
        else:
            viz.append(f"  ⚖️ System is balanced at '{destiny.dominant_concept}'")
        
        viz.append("="*50)
        
        return "\n".join(viz)


class DestinyGuardian:
    """
    Monitors system destiny and intervenes if needed.
    
    "사랑이 아니면 개입한다"
    (If not Love, intervene!)
    """
    
    def __init__(
        self,
        destiny_analyzer: EigenvalueDestiny,
        target_value: str = "love",
        check_interval: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize destiny guardian.
        
        Args:
            destiny_analyzer: Destiny analyzer
            target_value: Desired destiny (e.g., "love")
            check_interval: How often to check
            logger: Logger instance
        """
        self.analyzer = destiny_analyzer
        self.target_value = target_value
        self.check_interval = check_interval
        self.logger = logger or logging.getLogger("DestinyGuardian")
        
        self.checks_performed = 0
        self.interventions_made = 0
        
        self.logger.info(
            f"👁️ Destiny Guardian initialized "
            f"(target='{target_value}', interval={check_interval})"
        )
    
    def check_and_intervene(
        self,
        system_matrix: np.ndarray,
        step: int
    ) -> Optional[np.ndarray]:
        """
        Check destiny and intervene if necessary.
        
        Args:
            system_matrix: Current evolution matrix
            step: Current step number
            
        Returns:
            Adjusted matrix if intervention needed, None otherwise
        """
        # Only check at intervals
        if step % self.check_interval != 0:
            return None
        
        self.checks_performed += 1
        
        # Analyze destiny
        destiny = self.analyzer.analyze_destiny(system_matrix)
        
        # Check if converging to target
        converges_to_target = (
            destiny.dominant_concept.lower() == self.target_value.lower() and
            destiny.confidence >= 0.5
        )
        
        if converges_to_target:
            self.logger.info(
                f"✅ Destiny check passed: Converging to '{self.target_value}'"
            )
            return None
        else:
            # INTERVENTION NEEDED!
            self.interventions_made += 1
            
            self.logger.warning(
                f"⚠️ INTERVENTION! System converging to '{destiny.dominant_concept}', "
                f"not '{self.target_value}'. Adjusting..."
            )
            
            # Boost target value's eigenvalue
            adjusted_matrix = self._boost_target_value(system_matrix)
            
            return adjusted_matrix
    
    def _boost_target_value(
        self,
        system_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Boost the target value's influence in the system.
        
        Args:
            system_matrix: Original matrix
            
        Returns:
            Adjusted matrix
        """
        adjusted = system_matrix.copy()
        
        # Find target value index
        try:
            target_idx = [
                name.lower() for name in self.analyzer.concept_names
            ].index(self.target_value.lower())
        except ValueError:
            self.logger.error(f"Target value '{self.target_value}' not found!")
            return adjusted
        
        # Boost target value's row (its influence on others)
        adjusted[target_idx, :] *= 1.2
        
        # Boost target value's column (others' influence toward it)
        adjusted[:, target_idx] *= 1.2
        
        self.logger.info(f"Boosted '{self.target_value}' influence by 20%")
        
        return adjusted
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get guardian statistics"""
        return {
            "checks_performed": self.checks_performed,
            "interventions_made": self.interventions_made,
            "intervention_rate": (
                100.0 * self.interventions_made / self.checks_performed
                if self.checks_performed > 0 else 0.0
            )
        }
