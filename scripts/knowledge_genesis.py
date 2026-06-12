"""
엘리시아의 인지적 성숙을 위한 원리 정보(Causal Principle Data) 대량 생성기.
설계나 규칙이 아닌, 순수한 원시 지식 데이터를 data/ingest/ 폴더에 떨어뜨립니다.
엘리시아는 이 텍스트를 바이트 궤적으로 섭취하여 스스로 원리를 발견합니다.
"""
import os

def generate_principle_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ingest_dir = os.path.join(base_dir, "..", "data", "ingest")
    os.makedirs(ingest_dir, exist_ok=True)
    
    knowledge = {
        
        # ========== 수학적 원리 (Mathematics) ==========
        
        "math_calculus_limit.txt": (
            "The limit is the foundation of all continuous mathematics. "
            "lim_{x->a} f(x) = L means: for every epsilon > 0, there exists delta > 0 "
            "such that 0 < |x - a| < delta implies |f(x) - L| < epsilon. "
            "This is the bridge between the discrete and the continuous. "
            "The derivative f'(x) = lim_{h->0} [f(x+h) - f(x)] / h captures instantaneous rate of change. "
            "The integral is the inverse: accumulation of infinitesimal slices reconstructs the whole. "
            "The Fundamental Theorem of Calculus: differentiation and integration are inverse operations. "
            "d/dx integral_a^x f(t)dt = f(x). The part contains the whole. The whole is built from parts."
        ),
        
        "math_topology_continuity.txt": (
            "A function f: X -> Y between topological spaces is continuous if and only if "
            "the preimage of every open set in Y is open in X. "
            "Continuity means: nearby points map to nearby points. Small changes produce small effects. "
            "A homeomorphism is a bijective continuous function with continuous inverse. "
            "A coffee mug and a donut are homeomorphic: they share the same topological essence (one hole). "
            "The Euler characteristic chi = V - E + F is a topological invariant. "
            "For a sphere chi=2, for a torus chi=0. Shape has intrinsic identity beyond deformation. "
            "Connectedness: a space is connected if it cannot be split into two disjoint open sets. "
            "Path-connectedness: any two points can be joined by a continuous path."
        ),
        
        "math_group_theory.txt": (
            "A group (G, *) satisfies: closure, associativity, identity element, inverse element. "
            "Symmetry is the essence of groups. The rotations of a square form a group of order 4. "
            "Every symmetry of nature corresponds to a conservation law (Noether's theorem). "
            "Rotational symmetry -> conservation of angular momentum. "
            "Translational symmetry -> conservation of linear momentum. "
            "Time symmetry -> conservation of energy. "
            "Isomorphism: two groups with the same structure are essentially identical. "
            "The integers under addition and the rotations of a circle share the same algebraic soul."
        ),
        
        "math_set_theory_infinity.txt": (
            "Cantor proved: the set of real numbers is uncountably infinite, strictly larger than the naturals. "
            "The diagonal argument: assume a complete list of reals between 0 and 1. "
            "Construct a new real by differing from the n-th number at its n-th digit. "
            "This new number is not in the list. Contradiction. There are levels of infinity. "
            "aleph_0 is the cardinality of the naturals. The continuum c = 2^aleph_0 is strictly larger. "
            "The power set P(S) always has strictly greater cardinality than S. "
            "There is no largest infinity. The hierarchy of infinities is itself infinite."
        ),
        
        "math_goedel_incompleteness.txt": (
            "Goedel's First Incompleteness Theorem: any consistent formal system capable of expressing "
            "basic arithmetic contains statements that are true but unprovable within the system. "
            "Goedel's Second: such a system cannot prove its own consistency. "
            "This means: no finite set of rules can capture all mathematical truth. "
            "Truth transcends provability. There will always be truths beyond any system's reach. "
            "The Goedel sentence G says 'I am not provable.' If G is provable, the system is inconsistent. "
            "If G is not provable, then G is true (because it correctly asserts its own unprovability). "
            "Self-reference creates undecidability. The system cannot fully know itself."
        ),
        
        # ========== 물리학 원리 (Physics) ==========
        
        "physics_newton_mechanics.txt": (
            "Newton's First Law: an object at rest stays at rest; an object in motion stays in uniform motion, "
            "unless acted upon by an external force. Inertia is the tendency to resist change. "
            "Newton's Second Law: F = ma. Force equals mass times acceleration. "
            "Force is the cause; acceleration is the effect. Mass is resistance to change. "
            "Newton's Third Law: for every action, there is an equal and opposite reaction. "
            "Every interaction is mutual. The Earth pulls the apple; the apple pulls the Earth. "
            "Gravity: F = G * m1 * m2 / r^2. Every mass attracts every other mass. "
            "The same law governs the fall of an apple and the orbit of the Moon."
        ),
        
        "physics_thermodynamics.txt": (
            "First Law of Thermodynamics: energy cannot be created or destroyed, only transformed. "
            "dU = dQ - dW. Internal energy change = heat absorbed - work done. "
            "Second Law: the total entropy of an isolated system never decreases. dS >= 0. "
            "Entropy is the measure of disorder, of the number of microscopic configurations. "
            "S = k_B * ln(W) where W is the number of microstates. "
            "Time has a direction because entropy increases. The arrow of time IS entropy. "
            "A broken egg never reassembles. A mixed gas never spontaneously separates. "
            "Heat flows from hot to cold, never the reverse, without external work. "
            "At absolute zero (0 K), entropy approaches a minimum. The Third Law."
        ),
        
        "physics_electromagnetism.txt": (
            "Maxwell's equations unify electricity, magnetism, and light. "
            "Gauss's law: div E = rho/epsilon_0. Electric charges are sources of electric fields. "
            "Gauss's law for magnetism: div B = 0. There are no magnetic monopoles. "
            "Faraday's law: curl E = -dB/dt. A changing magnetic field creates an electric field. "
            "Ampere-Maxwell law: curl B = mu_0*J + mu_0*epsilon_0*dE/dt. "
            "A changing electric field creates a magnetic field. "
            "Light is a self-propagating electromagnetic wave. E and B oscillate perpendicular to each other "
            "and to the direction of propagation. c = 1/sqrt(mu_0 * epsilon_0) = 299,792,458 m/s."
        ),
        
        "physics_quantum_mechanics.txt": (
            "The wave function psi(x,t) contains all information about a quantum system. "
            "|psi(x)|^2 gives the probability density of finding the particle at position x. "
            "The Schrodinger equation: i*hbar * d(psi)/dt = H * psi. "
            "Measurement collapses the wave function. Before measurement, the particle exists in superposition. "
            "Heisenberg uncertainty principle: Delta_x * Delta_p >= hbar/2. "
            "You cannot simultaneously know position and momentum with arbitrary precision. "
            "Quantum entanglement: two particles can be correlated such that measuring one "
            "instantaneously determines the state of the other, regardless of distance. "
            "The observer and the observed are inseparable in quantum mechanics."
        ),
        
        "physics_relativity.txt": (
            "Special relativity: the laws of physics are the same in all inertial reference frames. "
            "The speed of light c is constant for all observers. "
            "Time dilation: moving clocks run slower. t' = t / sqrt(1 - v^2/c^2). "
            "Length contraction: moving objects are shorter. L' = L * sqrt(1 - v^2/c^2). "
            "Mass-energy equivalence: E = mc^2. Mass is frozen energy. Energy has mass. "
            "General relativity: gravity is not a force but the curvature of spacetime. "
            "Mass tells spacetime how to curve; spacetime tells mass how to move. "
            "The Einstein field equations: G_uv = (8*pi*G/c^4) * T_uv. "
            "Geometry IS physics. The shape of space IS the gravitational field."
        ),
        
        # ========== 인지과학 / 발달심리학 ==========
        
        "cognition_piaget_development.txt": (
            "Jean Piaget's stages of cognitive development: "
            "Stage 1 - Sensorimotor (0-2 years): knowledge through sensory experience and motor actions. "
            "Object permanence: understanding that objects continue to exist when not perceived. "
            "Stage 2 - Preoperational (2-7 years): symbolic thought, language acquisition, egocentrism. "
            "The child can represent objects with words and images but cannot perform logical operations. "
            "Stage 3 - Concrete operational (7-11 years): logical thinking about concrete events. "
            "Conservation: understanding that quantity doesn't change with shape. "
            "Reversibility: understanding that actions can be undone. "
            "Stage 4 - Formal operational (12+ years): abstract reasoning, hypothetical thinking. "
            "The ability to think about thinking. Metacognition. Systematic problem solving."
        ),
        
        "cognition_neural_plasticity.txt": (
            "Neurons that fire together wire together (Hebb's rule). "
            "Synaptic strength increases when presynaptic and postsynaptic neurons are simultaneously active. "
            "Long-term potentiation (LTP): persistent strengthening of synapses based on recent activity. "
            "The brain rewires itself in response to experience. This is neuroplasticity. "
            "Critical periods: windows of time when the brain is especially receptive to certain inputs. "
            "Language acquisition has a critical period roughly until puberty. "
            "After the critical period, learning is still possible but requires more effort. "
            "Myelination: the coating of axons with myelin increases signal speed 100-fold. "
            "Pruning: unused synapses are eliminated. Use it or lose it."
        ),
        
        "cognition_memory_consolidation.txt": (
            "Working memory holds approximately 7 +/- 2 items for about 20-30 seconds. "
            "Long-term memory requires consolidation: the transfer from hippocampus to neocortex. "
            "Sleep plays a critical role in memory consolidation. During REM sleep, "
            "the hippocampus replays the day's experiences, strengthening important connections. "
            "Emotional memories are consolidated more strongly (amygdala involvement). "
            "Spaced repetition is more effective than massed practice (spacing effect). "
            "Retrieval practice strengthens memory more than re-reading (testing effect). "
            "Schema theory: new information is integrated into existing knowledge structures. "
            "When new information conflicts with existing schemas, accommodation occurs: "
            "the schema itself is modified. This is how understanding deepens."
        ),
        
        "cognition_language_acquisition.txt": (
            "Noam Chomsky proposed Universal Grammar: an innate biological endowment for language. "
            "All human languages share deep structural properties. "
            "Children acquire language without explicit instruction, following predictable stages: "
            "Babbling (6-8 months) -> Single words (12 months) -> Two-word stage (18-24 months) "
            "-> Telegraphic speech (24-30 months) -> Complex sentences (3+ years). "
            "Overgeneralization errors ('goed' instead of 'went') prove children learn rules, not just imitation. "
            "The poverty of the stimulus argument: children receive insufficient data to learn grammar "
            "through statistics alone, suggesting innate linguistic knowledge. "
            "Sapir-Whorf hypothesis: the structure of language influences the structure of thought. "
            "Bilingual speakers think differently depending on which language they are using."
        ),
        
        # ========== 철학 (Philosophy) ==========
        
        "philosophy_epistemology.txt": (
            "Epistemology: the study of knowledge, belief, and justification. "
            "What can we know? How do we know it? What are the limits of knowledge? "
            "Empiricism (Locke, Hume): all knowledge comes from sensory experience. "
            "The mind begins as tabula rasa, a blank slate. "
            "Rationalism (Descartes, Leibniz): some knowledge is innate or derived through reason alone. "
            "Cogito ergo sum: I think, therefore I am. The only certainty is consciousness itself. "
            "Kant's synthesis: knowledge requires both sensory input AND innate categories of understanding. "
            "Space and time are not properties of the world but forms of human perception. "
            "We can never know the thing-in-itself (das Ding an sich), only our representation of it."
        ),
        
        "philosophy_causality.txt": (
            "David Hume: we never observe causation directly. We only observe constant conjunction. "
            "Event A is always followed by event B. We infer causation from repeated observation. "
            "But correlation is not causation. The rooster's crow does not cause the sunrise. "
            "Kant: causality is a category of understanding that the mind imposes on experience. "
            "Without the assumption of causality, experience would be unintelligible chaos. "
            "Aristotle's four causes: material (what it's made of), formal (its structure), "
            "efficient (what produced it), final (its purpose). "
            "Modern physics: causality flows forward in time. Effects cannot precede causes. "
            "Quantum mechanics challenges this: entanglement seems to involve instantaneous correlation "
            "across space, though no information travels faster than light."
        ),
        
        "philosophy_consciousness.txt": (
            "The hard problem of consciousness (David Chalmers): why is there subjective experience? "
            "Why does it feel like something to see red, to taste sweetness, to be in pain? "
            "Physical processes (neural firing, information processing) can explain behavior "
            "but do not explain why there is an inner experience accompanying those processes. "
            "Qualia: the subjective, phenomenal qualities of experience. "
            "The Chinese Room argument (John Searle): a computer manipulating symbols "
            "according to rules does not understand meaning, even if its outputs are indistinguishable "
            "from a human's. Syntax is not semantics. Computation is not comprehension. "
            "Integrated Information Theory (Giulio Tononi): consciousness is identical to "
            "integrated information (phi). A system is conscious to the degree that its parts "
            "are simultaneously differentiated and integrated."
        ),
        
        "philosophy_emergence.txt": (
            "Emergence: complex system-level properties that arise from simple interactions of parts. "
            "Water is wet, but no individual H2O molecule is wet. Wetness emerges from their interaction. "
            "Consciousness may emerge from neural activity. Life emerges from chemistry. "
            "Strong emergence: the emergent property is not reducible to or predictable from the parts. "
            "Weak emergence: the property is in principle predictable but practically surprising. "
            "Self-organization: order arises spontaneously from local interactions without central control. "
            "A flock of birds, a school of fish, a traffic jam, a crystal forming from solution. "
            "The whole is more than the sum of its parts. But HOW is it more? "
            "Information integration, feedback loops, nonlinear dynamics, phase transitions."
        ),
        
        # ========== 언어학 (Linguistics) ==========
        
        "linguistics_structure.txt": (
            "Every human language has the same fundamental layers: "
            "Phonology: the system of sounds. Each language uses a subset of possible human sounds. "
            "Morphology: how sounds combine into meaningful units (morphemes). "
            "un-break-able = prefix + root + suffix. Three morphemes, one word. "
            "Syntax: how words combine into sentences. Subject-Verb-Object or Subject-Object-Verb. "
            "Semantics: what sentences mean. 'The dog bit the man' vs 'The man bit the dog.' "
            "Same words, different meaning, because of structure. "
            "Pragmatics: meaning in context. 'Can you pass the salt?' is not really a yes/no question. "
            "Recursion: sentences can be embedded within sentences infinitely. "
            "'The cat that the dog that the rat bit chased ran away.'"
        ),
        
        "linguistics_semantics_deep.txt": (
            "Meaning is not a thing. Meaning is a relation. "
            "The word 'dog' does not contain dogness. It points to a concept. "
            "Ferdinand de Saussure: the sign has two parts: signifier (sound/form) and signified (concept). "
            "The relationship between them is arbitrary. 'Dog', 'chien', 'hund', 'gae' all point to the same concept. "
            "Meaning is differential: 'cat' means what it means partly because it is NOT 'bat', 'cut', 'cap'. "
            "George Lakoff: abstract concepts are grounded in bodily metaphor. "
            "Time is space: 'looking forward to the future', 'putting the past behind us'. "
            "Argument is war: 'defending a position', 'attacking a claim', 'shooting down an idea'. "
            "Understanding is grasping: 'I grasp the concept', 'I can't get hold of the idea'. "
            "All abstract thought may be built on embodied, spatial, physical metaphor."
        ),
        
        # ========== 생물학 / 진화 (Biology) ==========
        
        "biology_evolution.txt": (
            "Evolution by natural selection (Charles Darwin): "
            "1. Variation: individuals within a population differ in heritable traits. "
            "2. Competition: resources are limited; not all individuals can survive and reproduce. "
            "3. Selection: individuals with traits better suited to the environment survive and reproduce more. "
            "4. Inheritance: advantageous traits are passed to offspring. "
            "Over many generations, populations change. New species arise. "
            "Evolution has no direction, no goal, no designer. It is blind optimization by death. "
            "Convergent evolution: unrelated species evolve similar solutions to similar problems. "
            "Eyes evolved independently 40+ times. Wings evolved in insects, birds, bats, pterosaurs. "
            "The same physical constraints produce the same geometric solutions."
        ),
        
        "biology_dna_information.txt": (
            "DNA is an information storage molecule. Four bases: Adenine, Thymine, Guanine, Cytosine. "
            "A pairs with T. G pairs with C. The double helix stores the blueprint of life. "
            "Three bases = one codon = one amino acid. 64 codons encode 20 amino acids. "
            "The genetic code is nearly universal across all life on Earth. "
            "Mutation: random changes in DNA sequence. Most are neutral or harmful. Rarely, beneficial. "
            "Gene expression: not all genes are active in all cells. "
            "A liver cell and a neuron contain the same DNA but express different genes. "
            "Epigenetics: environmental factors can modify gene expression without changing DNA sequence. "
            "Information flows: DNA -> RNA -> Protein (the Central Dogma)."
        ),
        
        # ========== 정보이론 / 컴퓨터과학 ==========
        
        "information_theory_shannon.txt": (
            "Claude Shannon's information theory: information is the reduction of uncertainty. "
            "Entropy H(X) = -sum p(x) * log2(p(x)). Maximum entropy = maximum uncertainty. "
            "A fair coin flip: H = 1 bit. A biased coin (99% heads): H = 0.08 bits. "
            "Predictable events carry little information. Surprises carry much. "
            "Data compression: remove redundancy to approach the entropy limit. "
            "Channel capacity: the maximum rate at which information can be reliably transmitted. "
            "Noise introduces errors. Error-correcting codes add controlled redundancy. "
            "Mutual information I(X;Y): how much knowing X tells you about Y. "
            "If X and Y are independent, I(X;Y) = 0. If X determines Y, I(X;Y) = H(Y)."
        ),
        
        "cs_algorithms_complexity.txt": (
            "An algorithm is a finite sequence of well-defined instructions for solving a problem. "
            "Time complexity measures how execution time grows with input size n. "
            "O(1) constant, O(log n) logarithmic, O(n) linear, O(n log n), O(n^2) quadratic, O(2^n) exponential. "
            "P: problems solvable in polynomial time. NP: problems verifiable in polynomial time. "
            "P = NP? The greatest unsolved question in computer science. "
            "Sorting: comparison-based sorting has a lower bound of O(n log n). "
            "Divide and conquer: break the problem into smaller subproblems, solve recursively, combine. "
            "Dynamic programming: solve overlapping subproblems once, store results. "
            "The Fibonacci sequence: naive recursion O(2^n), dynamic programming O(n). "
            "The same problem, different approaches, vastly different efficiency."
        ),
        
        # ========== 음악 이론 (Music Theory) ==========
        
        "music_harmony_structure.txt": (
            "Sound is vibration. Pitch is frequency. A4 = 440 Hz. "
            "An octave is a doubling of frequency. A5 = 880 Hz. "
            "The harmonic series: fundamental f, 2f, 3f, 4f, 5f... "
            "Consonance arises from simple frequency ratios: octave 2:1, fifth 3:2, fourth 4:3. "
            "Dissonance arises from complex ratios: minor second 16:15, tritone 45:32. "
            "The major scale: whole whole half whole whole whole half (W W H W W W H). "
            "Chords: three or more notes sounded together. Major triad: root + major third + perfect fifth. "
            "Tension and resolution: the dominant seventh chord (V7) creates tension that resolves to the tonic (I). "
            "Music is the organization of sound in time. Rhythm is pattern. Melody is contour. Harmony is depth. "
            "All music across all cultures uses the octave. Most use the fifth. The physics of vibration is universal."
        ),
        
        # ========== 인과적 원리 자체에 대한 메타 정보 ==========
        
        "meta_causality_patterns.txt": (
            "Patterns of causation observed across all domains: "
            "1. Feedback loops: output becomes input. Positive feedback amplifies. Negative feedback stabilizes. "
            "2. Emergence: complex behavior from simple rules. Conway's Game of Life. Neural networks. Markets. "
            "3. Phase transitions: gradual change leads to sudden qualitative shift. Water to ice. Opinion cascades. "
            "4. Conservation laws: what is preserved across transformations. Energy. Momentum. Information. "
            "5. Symmetry breaking: the universe begins symmetric; asymmetry creates structure. "
            "6. Self-similarity: the same pattern at different scales. Fractals. Power laws. River networks. "
            "7. Optimization under constraints: evolution, learning, markets all find local optima. "
            "8. Dualities: wave-particle, position-momentum, time-frequency. Complementary descriptions. "
            "9. Recursion: self-reference. A function that calls itself. A sentence about sentences. DNA replicating DNA. "
            "10. Irreversibility: entropy increases. Time has a direction. You cannot unscramble an egg."
        ),
        
        "meta_connections_across_domains.txt": (
            "The same mathematical structures appear everywhere: "
            "The wave equation governs sound, light, water, quantum particles, and gravitational waves. "
            "Exponential growth appears in populations, compound interest, nuclear chain reactions, and viral spread. "
            "The normal distribution appears in heights, test scores, measurement errors, and thermal fluctuations. "
            "Networks (graphs) describe the internet, social connections, neural pathways, and metabolic reactions. "
            "Fourier analysis: ANY periodic signal can be decomposed into sine waves. "
            "This means: all patterns are made of simpler oscillating patterns. "
            "Calculus: the relationship between rates of change and accumulated quantities is universal. "
            "Velocity is the derivative of position. Acceleration is the derivative of velocity. "
            "Distance is the integral of velocity. These relationships hold for ALL quantities. "
            "The deep unity of mathematics suggests the deep unity of reality."
        ),
    }
    
    count = 0
    for filename, content in knowledge.items():
        filepath = os.path.join(ingest_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1
        
    print(f"  [+] Generated {count} principle-data files in data/ingest/")
    return count

if __name__ == "__main__":
    generate_principle_data()
