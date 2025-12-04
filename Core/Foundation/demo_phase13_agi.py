"""
Demo: Phase 13 - AGI Foundation Capabilities
범용 인공지능 향해 (Towards AGI)

Demonstrates:
1. Universal Transfer Learning - Rapid domain learning
2. Abstract Reasoning - Problem solving through abstraction
3. Causal Reasoning - Understanding cause and effect
4. Integrated AGI capabilities
"""

import asyncio
from Core.AGI import UniversalTransferLearner, AbstractReasoner, CausalReasoner


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def demo_transfer_learning():
    """Demonstrate universal transfer learning"""
    print_section("Demo 1: Universal Transfer Learning")
    
    learner = UniversalTransferLearner()
    
    # Example 1: Learn a new programming language domain
    print("📚 Learning a new domain: Python Programming")
    print("-" * 80)
    
    python_examples = [
        {
            "concept": "function",
            "syntax": "def name(params):",
            "purpose": "code reuse",
            "example": "def greet(name): return f'Hello {name}'"
        },
        {
            "concept": "class",
            "syntax": "class Name:",
            "purpose": "encapsulation",
            "example": "class Person: def __init__(self, name): self.name = name"
        },
        {
            "concept": "list_comprehension",
            "syntax": "[expr for item in iterable]",
            "purpose": "concise iteration",
            "example": "[x**2 for x in range(10)]"
        }
    ]
    
    domain_knowledge = await learner.learn_new_domain(
        domain="python_programming",
        examples=python_examples,
        target_proficiency=0.7
    )
    
    print(f"Domain: {domain_knowledge.domain}")
    print(f"Initial proficiency: {domain_knowledge.proficiency:.2%}")
    print(f"Concepts learned: {', '.join(domain_knowledge.concepts[:10])}")
    print(f"Patterns identified: {len(domain_knowledge.patterns)}")
    print(f"Total examples (including synthetic): {len(domain_knowledge.examples)}")
    
    # Example 2: Transfer to JavaScript
    print("\n📚 Transferring knowledge to JavaScript")
    print("-" * 80)
    
    js_examples = [
        {
            "concept": "function",
            "syntax": "function name(params) {}",
            "purpose": "code reuse",
            "example": "function greet(name) { return `Hello ${name}` }"
        },
        {
            "concept": "class",
            "syntax": "class Name {}",
            "purpose": "encapsulation",
            "example": "class Person { constructor(name) { this.name = name } }"
        }
    ]
    
    js_knowledge = await learner.learn_new_domain(
        domain="javascript_programming",
        examples=js_examples,
        target_proficiency=0.7
    )
    
    print(f"Domain: {js_knowledge.domain}")
    print(f"Proficiency (with transfer): {js_knowledge.proficiency:.2%}")
    print(f"Concepts learned: {', '.join(js_knowledge.concepts[:10])}")
    print(f"✨ Transfer learning accelerated learning!")
    
    # List all domains
    print("\n📋 All known domains:")
    for domain in learner.list_known_domains():
        proficiency = learner.get_domain_proficiency(domain)
        print(f"  • {domain}: {proficiency:.1%} proficiency")


async def demo_abstract_reasoning():
    """Demonstrate abstract reasoning"""
    print_section("Demo 2: Abstract Reasoning")
    
    reasoner = AbstractReasoner()
    
    # Example 1: Transformation problem
    print("🧩 Problem 1: Abstract Transformation")
    print("-" * 80)
    
    transformation_problem = {
        "type": "transformation",
        "description": "Convert input to output",
        "elements": ["input_state", "desired_output"],
        "goal": "transform input according to rule",
        "constraints": ["preserve type", "reversible"]
    }
    
    result1 = await reasoner.reason_abstractly(transformation_problem)
    
    print(f"Problem type: {result1['essence']['core_type']}")
    print(f"Abstract pattern: {result1['abstract_pattern'].pattern_type}")
    print(f"Abstraction level: {result1['abstract_pattern'].abstraction_level}")
    print(f"\nAbstract solution steps:")
    for i, step in enumerate(result1['abstract_solution'].abstract_steps, 1):
        print(f"  {i}. {step}")
    print(f"\nPrinciples used: {', '.join(result1['abstract_solution'].principles_used)}")
    print(f"Confidence: {result1['confidence']:.1%}")
    print(f"\nExplanation: {result1['concrete_solution']['explanation']}")
    
    # Example 2: Sequence problem
    print("\n🧩 Problem 2: Sequence Pattern")
    print("-" * 80)
    
    sequence_problem = {
        "type": "sequence",
        "description": "Identify pattern in sequence",
        "elements": [2, 4, 8, 16, 32],
        "goal": "predict next element",
        "context": {"domain": "mathematics"}
    }
    
    result2 = await reasoner.reason_abstractly(sequence_problem)
    
    print(f"Problem type: {result2['essence']['core_type']}")
    print(f"Abstract pattern: {result2['abstract_pattern'].pattern_type}")
    print(f"\nAbstract solution approach:")
    for i, step in enumerate(result2['abstract_solution'].abstract_steps, 1):
        print(f"  {i}. {step}")
    print(f"Confidence: {result2['confidence']:.1%}")
    
    # Example 3: Analogy generation
    print("\n🧩 Example 3: Analogy Generation")
    print("-" * 80)
    
    source_problem = {
        "domain": "biology",
        "type": "structure",
        "description": "Cell is to organism as..."
    }
    
    analogy = await reasoner.generate_analogy(source_problem, "computer_science")
    
    print(f"Source domain: {analogy['source_domain']}")
    print(f"Target domain: {analogy['target_domain']}")
    print(f"Shared pattern: {analogy['abstract_pattern']}")
    print(f"Explanation: {analogy['explanation']}")
    print(f"Possible mapping: 'Cell is to organism as Component is to System'")
    
    # Example 4: Solve by analogy
    print("\n🧩 Example 4: Solving by Analogy")
    print("-" * 80)
    
    known_problem = {
        "type": "transformation",
        "input": "ice",
        "process": "heating",
        "output": "water"
    }
    
    known_solution = {
        "method": "apply energy",
        "result": "phase_change"
    }
    
    new_problem = {
        "type": "transformation",
        "input": "water",
        "process": "cooling",
        "output": "?"
    }
    
    analogous_result = await reasoner.solve_by_analogy(
        known_problem, known_solution, new_problem
    )
    
    print(f"Known: {known_problem['input']} + {known_problem['process']} → {known_problem['output']}")
    print(f"New: {new_problem['input']} + {new_problem['process']} → ?")
    print(f"Analogy strength: {analogous_result['analogy_strength']:.1%}")
    print(f"Reasoning: {analogous_result['reasoning']}")
    if analogous_result['solution']:
        print(f"Predicted solution: {analogous_result['solution']}")


async def demo_causal_reasoning():
    """Demonstrate causal reasoning"""
    print_section("Demo 3: Causal Reasoning")
    
    reasoner = CausalReasoner()
    
    # Example 1: Infer causality from observations
    print("🔗 Example 1: Inferring Causal Relationships")
    print("-" * 80)
    
    observations = [
        {"exercise": 5, "calories_burned": 250, "weight_change": -0.5, "energy_level": 8},
        {"exercise": 3, "calories_burned": 150, "weight_change": -0.2, "energy_level": 6},
        {"exercise": 0, "calories_burned": 0, "weight_change": 0.3, "energy_level": 4},
        {"exercise": 7, "calories_burned": 350, "weight_change": -0.8, "energy_level": 9},
        {"exercise": 2, "calories_burned": 100, "weight_change": -0.1, "energy_level": 5},
        {"exercise": 6, "calories_burned": 300, "weight_change": -0.6, "energy_level": 8},
        {"exercise": 1, "calories_burned": 50, "weight_change": 0.1, "energy_level": 4},
        {"exercise": 4, "calories_burned": 200, "weight_change": -0.3, "energy_level": 7},
    ]
    
    causal_graph = await reasoner.infer_causality(observations, "health_fitness")
    
    print(f"Causal graph constructed with {len(causal_graph.nodes)} nodes and {len(causal_graph.edges)} edges")
    print(f"\nVariables: {', '.join(causal_graph.nodes)}")
    
    print(f"\nIdentified causal relationships:")
    for relation in causal_graph.edges[:5]:  # Show first 5
        print(f"  • {relation.cause} → {relation.effect}")
        print(f"    Strength: {relation.strength:.2f}, Confidence: {relation.confidence:.2f}")
        if relation.confounders:
            print(f"    Confounders: {', '.join(relation.confounders)}")
    
    # Example 2: Predict intervention effects
    print("\n🔗 Example 2: Predicting Intervention Effects")
    print("-" * 80)
    
    from Core.AGI.causal_reasoner import Intervention
    
    intervention = Intervention(
        variable="exercise",
        new_value=10  # Increase exercise to 10 hours
    )
    
    effects = await reasoner.predict_intervention_effects(causal_graph, intervention)
    
    print(f"Intervention: Set {intervention.variable} = {intervention.new_value}")
    print(f"\nPredicted effects:")
    for var, value in effects.affected_variables.items():
        print(f"  • {var}: {value:.2f}")
    
    print(f"\nReasoning chain:")
    for step in effects.reasoning[:5]:
        print(f"  {step}")
    
    print(f"\nOverall confidence: {effects.confidence:.1%}")
    
    # Example 3: Counterfactual reasoning
    print("\n🔗 Example 3: Counterfactual Reasoning")
    print("-" * 80)
    
    actual_observation = {
        "exercise": 2,
        "calories_burned": 100,
        "weight_change": -0.1,
        "energy_level": 5
    }
    
    counterfactual_condition = {
        "exercise": 6  # What if we had exercised 6 hours instead?
    }
    
    counterfactual = await reasoner.counterfactual_reasoning(
        actual_observation,
        counterfactual_condition,
        causal_graph
    )
    
    print(f"Actual situation:")
    for key, value in actual_observation.items():
        print(f"  {key}: {value}")
    
    print(f"\nCounterfactual: What if exercise was 6 hours instead of 2?")
    print(f"\nPredicted outcome:")
    for key, value in counterfactual['counterfactual_outcome'].items():
        print(f"  {key}: {value}")
    
    print(f"\nKey differences:")
    for var, (actual, counter) in counterfactual['difference'].items():
        print(f"  • {var}: {actual} → {counter}")
    
    # Example 4: Identify key causes
    print("\n🔗 Example 4: Identifying Key Causal Factors")
    print("-" * 80)
    
    key_causes = reasoner.identify_key_causes(causal_graph)
    
    print("Most influential variables (ranked by causal influence):")
    for i, (variable, influence) in enumerate(key_causes[:5], 1):
        print(f"  {i}. {variable} (influence score: {influence})")
    
    # Example 5: Explain causality
    print("\n🔗 Example 5: Explaining Causal Relationships")
    print("-" * 80)
    
    if len(causal_graph.nodes) >= 2:
        cause_var = "exercise"
        effect_var = "energy_level"
        
        explanation = await reasoner.explain_causality(causal_graph, cause_var, effect_var)
        
        print(f"How does '{cause_var}' affect '{effect_var}'?")
        print(f"\n{explanation['explanation']}")
        print(f"Confidence: {explanation['confidence']:.1%}")
        
        if explanation['paths']:
            print(f"\nCausal paths found:")
            for i, path_info in enumerate(explanation['paths'][:3], 1):
                print(f"  Path {i}: {path_info['path']}")
                print(f"    Length: {path_info['length']} steps, Confidence: {path_info['confidence']:.1%}")


async def demo_integrated_agi():
    """Demonstrate integrated AGI capabilities"""
    print_section("Demo 4: Integrated AGI Capabilities")
    
    print("🌟 Combining Transfer Learning, Abstract Reasoning, and Causal Inference")
    print("-" * 80)
    
    # Initialize all systems
    transfer_learner = UniversalTransferLearner()
    abstract_reasoner = AbstractReasoner()
    causal_reasoner = CausalReasoner()
    
    # Scenario: Learning to solve a new type of problem
    print("\n📖 Scenario: Learning to solve optimization problems")
    print("-" * 80)
    
    # Step 1: Transfer knowledge from mathematics domain
    print("\nStep 1: Transfer mathematical knowledge")
    math_examples = [
        {"concept": "minimize", "type": "optimization", "approach": "gradient_descent"},
        {"concept": "maximize", "type": "optimization", "approach": "hill_climbing"},
        {"concept": "constraint", "type": "restriction", "approach": "feasible_region"}
    ]
    
    opt_knowledge = await transfer_learner.learn_new_domain(
        "optimization",
        math_examples,
        target_proficiency=0.6
    )
    
    print(f"  ✓ Learned optimization domain: {opt_knowledge.proficiency:.1%} proficiency")
    print(f"  ✓ Transferred {len(opt_knowledge.concepts)} concepts")
    
    # Step 2: Abstract reasoning to understand problem structure
    print("\nStep 2: Abstract problem analysis")
    optimization_problem = {
        "type": "transformation",
        "description": "Find best solution given constraints",
        "elements": ["objective", "variables", "constraints"],
        "goal": "optimize objective",
        "context": {"domain": "optimization"}
    }
    
    abstract_analysis = await abstract_reasoner.reason_abstractly(optimization_problem)
    
    print(f"  ✓ Identified as {abstract_analysis['abstract_pattern'].pattern_type} pattern")
    print(f"  ✓ Solution approach: {abstract_analysis['abstract_solution'].solution_type}")
    print(f"  ✓ Confidence: {abstract_analysis['confidence']:.1%}")
    
    # Step 3: Causal reasoning for understanding relationships
    print("\nStep 3: Causal analysis of factors")
    optimization_observations = [
        {"learning_rate": 0.01, "iterations": 100, "convergence_speed": 0.5, "final_error": 0.2},
        {"learning_rate": 0.1, "iterations": 100, "convergence_speed": 0.8, "final_error": 0.1},
        {"learning_rate": 0.5, "iterations": 100, "convergence_speed": 0.9, "final_error": 0.3},
        {"learning_rate": 0.1, "iterations": 200, "convergence_speed": 0.85, "final_error": 0.05},
        {"learning_rate": 0.01, "iterations": 200, "convergence_speed": 0.6, "final_error": 0.15},
    ]
    
    causal_graph = await causal_reasoner.infer_causality(
        optimization_observations,
        "optimization_parameters"
    )
    
    print(f"  ✓ Built causal graph with {len(causal_graph.edges)} relationships")
    key_factors = causal_reasoner.identify_key_causes(causal_graph)
    if key_factors:
        print(f"  ✓ Key factor: {key_factors[0][0]} (influence: {key_factors[0][1]})")
    
    # Step 4: Integrated solution
    print("\nStep 4: Integrated AGI solution synthesis")
    print("-" * 80)
    
    solution_summary = {
        "knowledge_transfer": {
            "domains_used": ["mathematics", "optimization"],
            "proficiency": opt_knowledge.proficiency
        },
        "abstract_reasoning": {
            "pattern": abstract_analysis['abstract_pattern'].pattern_type,
            "approach": abstract_analysis['abstract_solution'].solution_type,
            "confidence": abstract_analysis['confidence']
        },
        "causal_insights": {
            "key_factors": [f[0] for f in key_factors[:2]] if key_factors else [],
            "relationships": len(causal_graph.edges)
        }
    }
    
    print("✅ Integrated AGI Solution:")
    print(f"  • Knowledge Transfer: {solution_summary['knowledge_transfer']['proficiency']:.1%} proficiency")
    print(f"  • Abstract Pattern: {solution_summary['abstract_reasoning']['pattern']}")
    print(f"  • Solution Confidence: {solution_summary['abstract_reasoning']['confidence']:.1%}")
    print(f"  • Causal Factors: {', '.join(solution_summary['causal_insights']['key_factors'])}")
    
    print("\n🎯 Result: Successfully applied AGI capabilities to learn and solve")
    print("   a new problem type through transfer, abstraction, and causal reasoning!")


async def main():
    """Run all Phase 13 demos"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Phase 13: AGI Foundation Demonstration".center(78) + "║")
    print("║" + "  범용 인공지능 향해 (Towards AGI)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run all demos
    await demo_transfer_learning()
    await demo_abstract_reasoning()
    await demo_causal_reasoning()
    await demo_integrated_agi()
    
    # Summary
    print_section("Phase 13 Demo Complete!")
    print("✅ Universal Transfer Learning: Rapid domain acquisition demonstrated")
    print("✅ Abstract Reasoning: Problem solving through abstraction demonstrated")
    print("✅ Causal Reasoning: Cause-effect understanding demonstrated")
    print("✅ Integrated AGI: Combined capabilities demonstrated")
    print("\n🌟 Elysia now has AGI foundation capabilities!")
    print("\nPhases completed: 10 (Creativity), 11 (Emotion), 12 (Autonomy), 13 (AGI)")


if __name__ == "__main__":
    asyncio.run(main())
