try:
    from .cognitive_engine import ElysiaCognitiveEngine
    from .field import CrystallizationField
    from .reflection_engram_engine import ReflectionEngram, GroundingTensionSensor, ReflectionEngramEngine
    from .wisdom_database_engine import WisdomDatabaseEngine
except ModuleNotFoundError:
    pass

from .inverse_mechanism_engine import (
    BoundaryCondition,
    ObservedTrajectory,
    DifferentialDelta,
    GeneratingMechanism,
    InverseMechanismEngine
)

from .non_tensor_meta_boundary import (
    SymmetryState,
    TypeConstraint,
    AxiomaticRelation,
    SymbolicTopologicalProof,
    StaticBypassManager
)

from .topological_axiomatic_engine import (
    MetaMechanismSignature,
    TopologicalAxiomaticEngine
)

from .causal_reframing_engine import (
    RawObservationLog,
    DeconstructedCausalStructure,
    CausalReframingEngine
)

try:
    from .topological_volumetric_architecture import (
        VolumetricPolytope,
        TopologicalSpaceEngine,
        SpacetimeTensorLayer4D,
        DynamicTopologicalRelaxationEngine,
        benchmark_flash_attention,
        benchmark_4d_spacetime_tensor
    )

    from .topological_rk4_autograd import (
        TopologicalRK4Function,
        TopologicalRK4Layer
    )

    from .predictive_coding import (
        PredictiveCodingNet
    )
except ModuleNotFoundError:
    pass

try:
    from .continuous_attractor_field import (
        ContinuousAttractorField
    )
except ModuleNotFoundError:
    pass

from .causal_phase_transition_engine import (
    GroundNode,
    GroundBeam,
    PerturbationWave,
    CausalProcessBlueprint,
    EpistemologicalReflectionRecord,
    ComplexImpedance,
    HomologyMetrics,
    CausalPhaseTransitionEngine
)

from .self_codification_engine import (
    FilteringLens,
    SelfCodificationRecord,
    SelfCodificationEngine
)

from .causal_boundary_tensor import CausalBoundaryTensor

from .causal_grounding_pipeline import (
    MinimalCausalEngine,
    ArchetypalCognitionEngine,
    SensoryInvariantModeling,
    DifferentialPerceptualEngine,
    CausalGroundingPipeline
)

from .lightweight_sparse_csm import (
    SparseCognitiveState,
    LightCognitiveStateMachine
)

from .phase_rectification import (
    PhaseRectificationFunction
)

from .cpt_causal_machine import (
    PhaseTransitionCausalMachine
)

from .sparse_hebbian_learner import (
    SparseHebbianCausalLearner
)

from .hierarchical_hebbian_engine import (
    HierarchicalHebbianCausalEngine
)

from .plasticity_memory_architecture import (
    CausalNode,
    CausalEdge,
    MetaCognitiveObservation,
    FrictionSensor,
    NodeAutopoiesis,
    GraphMutator,
    ConsolidationLoop,
    PlasticityMemoryArchitecture
)

from .topological_isomorphism_engine import (
    SubstrateStrainPoint,
    PhysicalConductanceBeam,
    MacroPhaseOrder,
    NonSymbolicReceptiveRefractor,
    TopologicalIsomorphismEngine,
    # Aliases for backwards compatibility where needed
    SubstrateStrainPoint as CausalLineageNode,
    PhysicalConductanceBeam as CausalLineageEdge,
    MacroPhaseOrder as MacroAxiom,
    NonSymbolicReceptiveRefractor as DomainReceptiveLens,
)
