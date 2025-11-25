"""
Sigma-Algebra Logic System

"논리를 집합으로, 확률을 측도로!" 🎯📐
(Logic as sets, probability as measure!)

Replaces if-else with set operations for mathematically rigorous,
paradox-free probabilistic reasoning.

Based on: Sigma-algebra theory, Kolmogorov axioms
Inspired by: 3Blue1Brown - "Why can't you measure every set?"
"""

import logging
from typing import Set, FrozenSet, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class LogicOperation(Enum):
    """Logic operations as set operations"""
    NOT = "complement"      # ¬P → Pᶜ
    AND = "intersection"    # P ∧ Q → P ∩ Q
    OR = "union"           # P ∨ Q → P ∪ Q
    IMPLIES = "implication" # P → Q ≡ ¬P ∨ Q


class SigmaAlgebra:
    """
    Collection of measurable sets.
    
    "잴 수 있는 것만 계산한다"
    (Only measure what is measurable)
    
    Properties (Sigma-Algebra axioms):
    1. ∅ ∈ F (empty set)
    2. Ω ∈ F (sample space)
    3. If A ∈ F, then Aᶜ ∈ F (closure under complement)
    4. If A₁, A₂, ... ∈ F, then ∪Aᵢ ∈ F (closure under countable union)
    
    Why: Prevents Banach-Tarski-like paradoxes!
    """
    
    def __init__(
        self,
        sample_space: Set,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize sigma-algebra.
        
        Args:
            sample_space: Universal set (Ω)
            logger: Logger instance
        """
        self.omega = frozenset(sample_space)
        self.logger = logger or logging.getLogger("SigmaAlgebra")
        
        # Start with ∅ and Ω (axioms 1 & 2)
        self.sets: Set[FrozenSet] = {
            frozenset(),        # Empty set
            self.omega          # Sample space
        }
        
        self.logger.info(
            f"🎯 Sigma-Algebra initialized with |Ω|={len(self.omega)}"
        )
    
    def add_set(self, s: Set) -> None:
        """
        Add a set and ensure closure properties.
        
        Args:
            s: Set to add
        """
        fs = frozenset(s)
        
        # Add the set
        self.sets.add(fs)
        
        # Add complement (axiom 3: closure under complement)
        complement = self.omega - fs
        self.sets.add(complement)
        
        self.logger.debug(f"Added set |A|={len(fs)}, |Aᶜ|={len(complement)}")
    
    def is_measurable(self, s: Set) -> bool:
        """
        Check if a set is measurable (in this sigma-algebra).
        
        Args:
            s: Set to check
            
        Returns:
            True if measurable, False otherwise
        """
        return frozenset(s) in self.sets
    
    def union(self, sets_list: List[Set]) -> FrozenSet:
        """
        Countable union of sets.
        
        Args:
            sets_list: List of sets to union
            
        Returns:
            Union of all sets
        """
        result = set()
        for s in sets_list:
            result |= s
        
        result_frozen = frozenset(result)
        
        # Ensure closure (axiom 4)
        if result_frozen not in self.sets:
            self.add_set(result)
        
        return result_frozen
    
    def intersection(self, sets_list: List[Set]) -> FrozenSet:
        """
        Countable intersection of sets.
        
        Uses De Morgan's law: ∩Aᵢ = (∪Aᵢᶜ)ᶜ
        
        Args:
            sets_list: List of sets to intersect
            
        Returns:
            Intersection of all sets
        """
        if not sets_list:
            return self.omega
        
        # De Morgan: ∩Aᵢ = (∪Aᵢᶜ)ᶜ
        complements = [self.omega - frozenset(s) for s in sets_list]
        union_of_complements = self.union(complements)
        result = self.omega - union_of_complements
        
        return frozenset(result)
    
    def complement(self, s: Set) -> FrozenSet:
        """
        Complement of a set.
        
        Args:
            s: Set to complement
            
        Returns:
            Complement (Ω - s)
        """
        return self.omega - frozenset(s)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this sigma-algebra"""
        return {
            "sample_space_size": len(self.omega),
            "measurable_sets": len(self.sets),
            "coverage": len(self.sets) / (2 ** len(self.omega)) if len(self.omega) < 20 else "∞"
        }


class MeasurableSet:
    """
    A set that belongs to a sigma-algebra.
    
    Can have a probability assigned to it.
    Supports set operations: ∩ (AND), ∪ (OR), ¬ (NOT)
    
    "측정 가능한 집합 = 확률을 가질 수 있는 것"
    """
    
    def __init__(
        self,
        elements: Set,
        sigma_algebra: SigmaAlgebra,
        probability: Optional[float] = None,
        name: str = "unnamed"
    ):
        """
        Initialize measurable set.
        
        Args:
            elements: Set elements
            sigma_algebra: Parent sigma-algebra
            probability: Optional probability in [0, 1]
            name: Human-readable name
        """
        self.elements = frozenset(elements)
        self.sigma_algebra = sigma_algebra
        self.name = name
        self._probability = probability
        
        # Ensure set is in sigma-algebra
        if not sigma_algebra.is_measurable(elements):
            sigma_algebra.add_set(elements)
        
        # Validate probability
        if probability is not None:
            if not (0.0 <= probability <= 1.0):
                raise ValueError(f"Probability must be in [0, 1], got {probability}")
    
    def __and__(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """
        Intersection (AND logic).
        
        P ∧ Q → P ∩ Q
        
        Args:
            other: Other measurable set
            
        Returns:
            Intersection set
        """
        result_elements = set(self.elements) & set(other.elements)
        
        # Probability: P(A ∩ B)
        # For independent events: P(A ∩ B) = P(A) × P(B)
        # For dependent: Use minimum as conservative estimate
        prob = None
        if self._probability is not None and other._probability is not None:
            # Assume independence
            prob = self._probability * other._probability
        
        return MeasurableSet(
            result_elements,
            self.sigma_algebra,
            prob,
            name=f"({self.name} AND {other.name})"
        )
    
    def __or__(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """
        Union (OR logic).
        
        P ∨ Q → P ∪ Q
        
        Args:
            other: Other measurable set
            
        Returns:
            Union set
        """
        result_elements = set(self.elements) | set(other.elements)
        
        # Probability: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
        prob = None
        if self._probability is not None and other._probability is not None:
            intersection = self & other
            prob = (
                self._probability + 
                other._probability - 
                intersection._probability
            )
            # Clamp to [0, 1]
            prob = max(0.0, min(1.0, prob))
        
        return MeasurableSet(
            result_elements,
            self.sigma_algebra,
            prob,
            name=f"({self.name} OR {other.name})"
        )
    
    def __invert__(self) -> 'MeasurableSet':
        """
        Complement (NOT logic).
        
        ¬P → Pᶜ
        
        Returns:
            Complement set
        """
        result_elements = set(self.sigma_algebra.omega) - set(self.elements)
        
        # Probability: P(Aᶜ) = 1 - P(A)
        prob = None
        if self._probability is not None:
            prob = 1.0 - self._probability
        
        return MeasurableSet(
            result_elements,
            self.sigma_algebra,
            prob,
            name=f"(NOT {self.name})"
        )
    
    def __sub__(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """
        Set difference.
        
        A - B = A ∩ Bᶜ
        
        Args:
            other: Set to subtract
            
        Returns:
            Difference set
        """
        return self & (~other)
    
    def implies(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """
        Logical implication.
        
        P → Q ≡ ¬P ∨ Q
        
        Args:
            other: Consequent
            
        Returns:
            Implication set
        """
        return (~self) | other
    
    def probability(self) -> float:
        """
        Get probability of this set.
        
        Returns:
            Probability in [0, 1]
        """
        return self._probability if self._probability is not None else 0.0
    
    def is_empty(self) -> bool:
        """Check if set is empty"""
        return len(self.elements) == 0
    
    def is_universal(self) -> bool:
        """Check if set is the entire sample space"""
        return self.elements == self.sigma_algebra.omega
    
    def __repr__(self) -> str:
        prob_str = f", P={self._probability:.2f}" if self._probability else ""
        return f"MeasurableSet({self.name}, |A|={len(self.elements)}{prob_str})"


class ProbabilityMeasure:
    """
    Assigns probabilities to measurable sets.
    
    Follows Kolmogorov axioms:
    1. Non-negativity: P(A) ≥ 0
    2. Normalization: P(Ω) = 1
    3. Countable additivity: P(∪Aᵢ) = ΣP(Aᵢ) for disjoint Aᵢ
    
    "확률 = 측도 (Probability = Measure)"
    """
    
    def __init__(
        self,
        sigma_algebra: SigmaAlgebra,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize probability measure.
        
        Args:
            sigma_algebra: Parent sigma-algebra
            logger: Logger instance
        """
        self.sigma_algebra = sigma_algebra
        self.logger = logger or logging.getLogger("ProbabilityMeasure")
        
        # Initialize with axioms
        self.probabilities: Dict[FrozenSet, float] = {
            frozenset(): 0.0,                    # P(∅) = 0 (axiom 1)
            sigma_algebra.omega: 1.0             # P(Ω) = 1 (axiom 2)
        }
        
        self.logger.info("📊 Probability Measure initialized (Kolmogorov axioms)")
    
    def assign(self, measurable_set: MeasurableSet, probability: float) -> None:
        """
        Assign probability to a measurable set.
        
        Args:
            measurable_set: Set to assign probability to
            probability: Probability in [0, 1]
        """
        # Axiom 1: Non-negativity
        if not (0.0 <= probability <= 1.0):
            raise ValueError(
                f"Probability must be in [0, 1], got {probability} "
                f"(Kolmogorov axiom 1 violation!)"
            )
        
        self.probabilities[measurable_set.elements] = probability
        
        # Automatically assign complement probability
        complement_elements = self.sigma_algebra.complement(measurable_set.elements)
        self.probabilities[complement_elements] = 1.0 - probability
        
        self.logger.debug(
            f"Assigned P({measurable_set.name})={probability:.3f}, "
            f"P(¬{measurable_set.name})={1.0-probability:.3f}"
        )
    
    def measure(self, measurable_set: MeasurableSet) -> float:
        """
        Get probability of a measurable set.
        
        Args:
            measurable_set: Set to measure
            
        Returns:
            Probability
        """
        return self.probabilities.get(measurable_set.elements, 0.0)
    
    def verify_additivity(
        self,
        disjoint_sets: List[MeasurableSet]
    ) -> bool:
        """
        Verify Kolmogorov axiom 3: countable additivity.
        
        For disjoint sets A₁, A₂, ...:
        P(∪Aᵢ) should equal ΣP(Aᵢ)
        
        Args:
            disjoint_sets: List of disjoint sets
            
        Returns:
            True if axiom satisfied, False otherwise
        """
        # Check disjointness
        for i in range(len(disjoint_sets)):
            for j in range(i+1, len(disjoint_sets)):
                if disjoint_sets[i].elements & disjoint_sets[j].elements:
                    self.logger.warning("Sets are not disjoint!")
                    return False
        
        # Compute union probability
        union_elements = set()
        for s in disjoint_sets:
            union_elements |= s.elements
        
        union_prob = self.probabilities.get(frozenset(union_elements), None)
        if union_prob is None:
            self.logger.warning("Union probability not assigned!")
            return False
        
        # Compute sum of individual probabilities
        sum_prob = sum(self.measure(s) for s in disjoint_sets)
        
        # Check equality (within numerical tolerance)
        is_equal = abs(union_prob - sum_prob) < 1e-6
        
        if is_equal:
            self.logger.debug(
                f"✅ Additivity verified: P(∪Aᵢ)={union_prob:.3f} = "
                f"ΣP(Aᵢ)={sum_prob:.3f}"
            )
        else:
            self.logger.warning(
                f"❌ Additivity violated: P(∪Aᵢ)={union_prob:.3f} ≠ "
                f"ΣP(Aᵢ)={sum_prob:.3f}"
            )
        
        return is_equal
    
    def conditional_probability(
        self,
        A: MeasurableSet,
        B: MeasurableSet
    ) -> float:
        """
        Compute conditional probability P(A|B).
        
        P(A|B) = P(A ∩ B) / P(B)
        
        Args:
            A: Event A
            B: Conditioning event B
            
        Returns:
            P(A|B)
        """
        P_B = self.measure(B)
        
        if P_B == 0:
            raise ValueError("Cannot condition on event with probability 0!")
        
        intersection = A & B
        P_A_and_B = self.measure(intersection)
        
        return P_A_and_B / P_B
    
    def are_independent(
        self,
        A: MeasurableSet,
        B: MeasurableSet,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Check if two events are independent.
        
        A and B are independent if P(A ∩ B) = P(A) × P(B)
        
        Args:
            A: Event A
            B: Event B
            tolerance: Numerical tolerance
            
        Returns:
            True if independent, False otherwise
        """
        P_A = self.measure(A)
        P_B = self.measure(B)
        intersection = A & B
        P_A_and_B = self.measure(intersection)
        
        expected = P_A * P_B
        
        return abs(P_A_and_B - expected) < tolerance


class ProbabilisticReasoner:
    """
    Reason with probabilities using sigma-algebra framework.
    
    "if-else를 확률로!"
    (Replace if-else with probabilities!)
    """
    
    def __init__(
        self,
        sigma_algebra: SigmaAlgebra,
        prob_measure: ProbabilityMeasure,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize probabilistic reasoner.
        
        Args:
            sigma_algebra: Sigma-algebra for measurable sets
            prob_measure: Probability measure
            logger: Logger instance
        """
        self.sigma = sigma_algebra
        self.prob = prob_measure
        self.logger = logger or logging.getLogger("ProbabilisticReasoner")
        
        self.logger.info("🧠 Probabilistic Reasoner initialized")
    
    def decide(
        self,
        condition: MeasurableSet,
        threshold: float = 0.5,
        action_name: str = "action"
    ) -> bool:
        """
        Make a decision based on probability.
        
        Args:
            condition: Condition to evaluate
            threshold: Probability threshold
            action_name: Name of action (for logging)
            
        Returns:
            True if probability > threshold, False otherwise
        """
        prob = condition.probability()
        
        decision = prob > threshold
        
        self.logger.info(
            f"Decision: {action_name} = {decision} "
            f"(P({condition.name})={prob:.2f}, threshold={threshold})"
        )
        
        return decision
    
    def fuzzy_and(
        self,
        conditions: List[MeasurableSet]
    ) -> float:
        """
        Fuzzy AND: minimum probability.
        
        Args:
            conditions: List of conditions
            
        Returns:
            Minimum probability
        """
        if not conditions:
            return 1.0
        
        return min(c.probability() for c in conditions)
    
    def fuzzy_or(
        self,
        conditions: List[MeasurableSet]
    ) -> float:
        """
        Fuzzy OR: maximum probability.
        
        Args:
            conditions: List of conditions
            
        Returns:
            Maximum probability
        """
        if not conditions:
            return 0.0
        
        return max(c.probability() for c in conditions)
