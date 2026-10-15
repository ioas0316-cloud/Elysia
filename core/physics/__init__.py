"""Elysia Core - Physics / Kinematics Subsystem"""

try:
    from core.physics.counterfactual_branching import CounterfactualBranchingEngine
except ImportError:
    CounterfactualBranchingEngine = None

try:
    from core.physics.frame_controller import ObservationalFrameController
except ImportError:
    ObservationalFrameController = None

try:
    from core.physics.constructive_causal_spacetime import (
        ConstructiveSpacetimeAxis,
        ConstructiveLogicDiscriminator,
        HierarchicalScaleCoupler,
    )
except ImportError:
    ConstructiveSpacetimeAxis = None
    ConstructiveLogicDiscriminator = None
    HierarchicalScaleCoupler = None

try:
    from core.physics.quantum_rotor_phase import (
        Clifford5DRotorEngine,
        LandauGinzburgPotentialEngine,
        QuantumRotorPhaseIntegrator
    )
except ImportError:
    Clifford5DRotorEngine = None
    LandauGinzburgPotentialEngine = None
    QuantumRotorPhaseIntegrator = None

try:
    from core.physics.tensor_helmholtz_decomposer import (
        TensorHelmholtzMetricDecomposer,
        MetricWaveDecompositionResult
    )
except ImportError:
    TensorHelmholtzMetricDecomposer = None
    MetricWaveDecompositionResult = None

try:
    from core.physics.metric_rotor_pipeline import (
        IntegratedMetricRotorPipeline,
        MetricToRotorBridge
    )
except ImportError:
    IntegratedMetricRotorPipeline = None
    MetricToRotorBridge = None

try:
    from core.physics.trainable_metric_rotor_loop import TrainableMetricRotorPipeline
except ImportError:
    TrainableMetricRotorPipeline = None

try:
    from core.physics.waveform_scale_protocol import (
        WaveformScaleProtocolEngine,
        MicroScaleCluster,
        PhaseLockingMetrics,
    )
except ImportError:
    WaveformScaleProtocolEngine = None
    MicroScaleCluster = None
    PhaseLockingMetrics = None

try:
    from core.physics.cosmic_grounding_protocol import (
        UnifiedCosmicSpectrumEngine,
        SpectrumFieldState,
        CosmicGroundingMetrics,
        HolographicFieldRenderer,
        SynestheticSpectrumMapper,
        DiscriminativeSensoryObservationLens,
    )
except ImportError:
    UnifiedCosmicSpectrumEngine = None
    SpectrumFieldState = None
    CosmicGroundingMetrics = None
    HolographicFieldRenderer = None
    SynestheticSpectrumMapper = None
    DiscriminativeSensoryObservationLens = None

try:
    from core.physics.conceptual_causal_tensor_engine import ConceptualCausalTensorEngine
except ImportError:
    ConceptualCausalTensorEngine = None

try:
    from core.physics.arc_reactor_drive_axis import ArcReactorDriveEngine, LocalZeroFieldState
except ImportError:
    ArcReactorDriveEngine = None
    LocalZeroFieldState = None

try:
    from core.physics.isomorphic_phase_field import (
        IsomorphicPhaseFieldSubstrate,
        ModalitySignature,
        ScaleBoundaryResult,
    )
except ImportError:
    IsomorphicPhaseFieldSubstrate = None
    ModalitySignature = None
    ScaleBoundaryResult = None

__all__ = [
    "ArcReactorDriveEngine",
    "LocalZeroFieldState",
]

for name in [
    "CounterfactualBranchingEngine",
    "ObservationalFrameController",
    "ConstructiveSpacetimeAxis",
    "ConstructiveLogicDiscriminator",
    "HierarchicalScaleCoupler",
    "Clifford5DRotorEngine",
    "LandauGinzburgPotentialEngine",
    "QuantumRotorPhaseIntegrator",
    "TensorHelmholtzMetricDecomposer",
    "MetricWaveDecompositionResult",
    "IntegratedMetricRotorPipeline",
    "MetricToRotorBridge",
    "TrainableMetricRotorPipeline",
    "WaveformScaleProtocolEngine",
    "MicroScaleCluster",
    "PhaseLockingMetrics",
    "UnifiedCosmicSpectrumEngine",
    "SpectrumFieldState",
    "CosmicGroundingMetrics",
    "HolographicFieldRenderer",
    "SynestheticSpectrumMapper",
    "DiscriminativeSensoryObservationLens",
    "ConceptualCausalTensorEngine",
    "IsomorphicPhaseFieldSubstrate",
    "ModalitySignature",
    "ScaleBoundaryResult",
]:
    if globals().get(name) is not None:
        __all__.append(name)
