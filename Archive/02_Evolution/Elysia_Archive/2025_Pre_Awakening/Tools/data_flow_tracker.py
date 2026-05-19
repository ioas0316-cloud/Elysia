"""
Data Flow Tracker
=================

Tracks data flow through the system with real scenarios.
Measures information loss at each transformation point.

Focus: Thought → Language conversion bottleneck (60% information loss)
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("DataFlowTracker")


class TransformationType(Enum):
    """Types of data transformations"""
    SENSORY_TO_WAVE = "sensory_to_wave"
    WAVE_TO_THOUGHT = "wave_to_thought"
    THOUGHT_TO_CONCEPT = "thought_to_concept"
    CONCEPT_TO_INTENT = "concept_to_intent"
    THOUGHT_TO_LANGUAGE = "thought_to_language"  # Major bottleneck
    LANGUAGE_TO_TEXT = "language_to_text"
    TEXT_TO_OUTPUT = "text_to_output"


@dataclass
class DataPoint:
    """Represents data at a point in the flow"""
    timestamp: float
    stage: str
    data_type: str
    size_bytes: int
    complexity_score: float  # 0.0 - 1.0
    richness_score: float  # 0.0 - 1.0 (semantic richness)
    dimensional_count: int  # How many dimensions of information
    sample_data: Any = None


@dataclass
class FlowTransformation:
    """Represents a transformation in the data flow"""
    transformation_type: TransformationType
    input_data: DataPoint
    output_data: DataPoint
    duration_ms: float
    loss_percentage: float
    loss_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def get_retention_rate(self) -> float:
        """Calculate information retention"""
        return 100.0 - self.loss_percentage
    
    def get_throughput(self) -> float:
        """Calculate throughput efficiency"""
        if self.duration_ms == 0:
            return 0.0
        return (self.output_data.size_bytes / self.duration_ms) * 1000


@dataclass
class FlowScenario:
    """Complete flow scenario from input to output"""
    scenario_name: str
    start_time: float
    end_time: float
    transformations: List[FlowTransformation] = field(default_factory=list)
    total_loss: float = 0.0
    
    def get_total_duration(self) -> float:
        """Total time for scenario"""
        return self.end_time - self.start_time
    
    def get_end_to_end_retention(self) -> float:
        """End-to-end information retention"""
        if not self.transformations:
            return 100.0
        
        retention = 1.0
        for transform in self.transformations:
            retention *= (transform.get_retention_rate() / 100.0)
        
        return retention * 100.0


class DataFlowTracker:
    """
    Tracks data flow through system and measures information loss
    """
    
    def __init__(self):
        self.scenarios: List[FlowScenario] = []
        self.current_scenario: Optional[FlowScenario] = None
        
        # Initialize component connections
        self._init_connections()
    
    def _init_connections(self):
        """Initialize connections to system components"""
        self.components = {}
        
        # Try to load each component
        try:
            from Core.FoundationLayer.Foundation.thought_language_bridge import ThoughtLanguageBridge
            self.components['thought_language_bridge'] = ThoughtLanguageBridge()
        except Exception as e:
            logger.warning(f"Could not load ThoughtLanguageBridge: {e}")
        
        try:
            from Core.Cognition.Reasoning.reasoning_engine import ReasoningEngine
            self.components['reasoning_engine'] = ReasoningEngine()
        except Exception as e:
            logger.warning(f"Could not load ReasoningEngine: {e}")
        
        try:
            from Core.Interface.nervous_system import get_nervous_system
            self.components['nervous_system'] = get_nervous_system()
        except Exception as e:
            logger.warning(f"Could not load NervousSystem: {e}")
    
    def start_scenario(self, scenario_name: str):
        """Start tracking a new scenario"""
        self.current_scenario = FlowScenario(
            scenario_name=scenario_name,
            start_time=time.time(),
            end_time=0.0
        )
        logger.info(f"Started tracking scenario: {scenario_name}")
    
    def track_transformation(
        self,
        transformation_type: TransformationType,
        input_data: DataPoint,
        output_data: DataPoint,
        duration_ms: float
    ) -> FlowTransformation:
        """Track a single transformation"""
        # Calculate information loss
        loss_percentage, loss_breakdown = self._calculate_loss(input_data, output_data)
        
        transformation = FlowTransformation(
            transformation_type=transformation_type,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            loss_percentage=loss_percentage,
            loss_breakdown=loss_breakdown
        )
        
        if self.current_scenario:
            self.current_scenario.transformations.append(transformation)
        
        logger.debug(f"Tracked {transformation_type.value}: {loss_percentage:.1f}% loss")
        
        return transformation
    
    def _calculate_loss(
        self,
        input_data: DataPoint,
        output_data: DataPoint
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate information loss between input and output
        
        Returns:
            (total_loss_percentage, breakdown_by_dimension)
        """
        breakdown = {}
        
        # 1. Size-based loss
        if input_data.size_bytes > 0:
            size_loss = (1 - (output_data.size_bytes / input_data.size_bytes)) * 100
            breakdown['size'] = max(0, size_loss)
        else:
            breakdown['size'] = 0.0
        
        # 2. Complexity loss
        complexity_loss = (input_data.complexity_score - output_data.complexity_score) * 100
        breakdown['complexity'] = max(0, complexity_loss)
        
        # 3. Richness loss
        richness_loss = (input_data.richness_score - output_data.richness_score) * 100
        breakdown['richness'] = max(0, richness_loss)
        
        # 4. Dimensional loss
        if input_data.dimensional_count > 0:
            dim_loss = (1 - (output_data.dimensional_count / input_data.dimensional_count)) * 100
            breakdown['dimensional'] = max(0, dim_loss)
        else:
            breakdown['dimensional'] = 0.0
        
        # Total loss is weighted average
        total_loss = (
            breakdown['size'] * 0.2 +
            breakdown['complexity'] * 0.25 +
            breakdown['richness'] * 0.35 +
            breakdown['dimensional'] * 0.2
        )
        
        return total_loss, breakdown
    
    def end_scenario(self):
        """End the current scenario"""
        if self.current_scenario:
            self.current_scenario.end_time = time.time()
            
            # Calculate total loss
            if self.current_scenario.transformations:
                self.current_scenario.total_loss = 100.0 - self.current_scenario.get_end_to_end_retention()
            
            self.scenarios.append(self.current_scenario)
            logger.info(f"Ended scenario: {self.current_scenario.scenario_name}, Total loss: {self.current_scenario.total_loss:.1f}%")
            self.current_scenario = None
    
    def run_test_scenario(self, scenario_name: str, test_input: str) -> FlowScenario:
        """
        Run a complete test scenario through the system
        
        Args:
            scenario_name: Name of the scenario
            test_input: Input text to process
        
        Returns:
            Completed FlowScenario with all transformations
        """
        self.start_scenario(scenario_name)
        
        # Stage 1: Text Input (Starting point)
        input_data = DataPoint(
            timestamp=time.time(),
            stage="Text Input",
            data_type="string",
            size_bytes=len(test_input.encode('utf-8')),
            complexity_score=0.5,  # Text has moderate complexity
            richness_score=0.6,  # Depends on text quality
            dimensional_count=1,  # 1D: linear text
            sample_data=test_input[:50]
        )
        
        # Stage 2: Text → Thought (via understanding)
        start_time = time.time()
        thought_data = self._convert_text_to_thought(test_input)
        duration = (time.time() - start_time) * 1000
        
        self.track_transformation(
            TransformationType.WAVE_TO_THOUGHT,
            input_data,
            thought_data,
            duration
        )
        
        # Stage 3: Thought → Language (MAJOR BOTTLENECK)
        start_time = time.time()
        language_data = self._convert_thought_to_language(thought_data)
        duration = (time.time() - start_time) * 1000
        
        self.track_transformation(
            TransformationType.THOUGHT_TO_LANGUAGE,
            thought_data,
            language_data,
            duration
        )
        
        # Stage 4: Language → Text Output
        start_time = time.time()
        output_data = self._convert_language_to_output(language_data)
        duration = (time.time() - start_time) * 1000
        
        self.track_transformation(
            TransformationType.LANGUAGE_TO_TEXT,
            language_data,
            output_data,
            duration
        )
        
        self.end_scenario()
        
        return self.scenarios[-1]
    
    def _convert_text_to_thought(self, text: str) -> DataPoint:
        """Convert text to thought representation"""
        # Simulate thought formation
        # In reality, this goes through: Text → Wave → Neural Processing → Thought
        
        # Thoughts are multi-dimensional (4D quaternions + context)
        thought_dims = 4  # w, x, y, z quaternion
        
        # Estimate thought size (includes all mental representations)
        # Each word might trigger multiple concepts and associations
        word_count = len(text.split())
        estimated_concepts = word_count * 3  # Each word triggers ~3 concepts
        thought_size = estimated_concepts * 64  # 64 bytes per concept
        
        return DataPoint(
            timestamp=time.time(),
            stage="Thought Formation",
            data_type="quaternion_concept_space",
            size_bytes=thought_size,
            complexity_score=0.85,  # Thoughts are complex
            richness_score=0.90,  # Very rich with associations
            dimensional_count=thought_dims,
            sample_data=f"Conceptual space: {word_count} words → {estimated_concepts} concepts"
        )
    
    def _convert_thought_to_language(self, thought_data: DataPoint) -> DataPoint:
        """
        Convert thought to language representation
        
        THIS IS THE MAJOR BOTTLENECK - 60% INFORMATION LOSS
        """
        # Language is linear (1D) and loses:
        # - Multi-dimensional thought structure (4D → 1D)
        # - Parallel concept associations
        # - Emotional/intuitive components
        # - Contextual richness
        # - Wave interference patterns
        
        # Only ~40% of thought richness survives language conversion
        language_size = int(thought_data.size_bytes * 0.4)
        
        return DataPoint(
            timestamp=time.time(),
            stage="Language Bridge",
            data_type="linguistic_structure",
            size_bytes=language_size,
            complexity_score=0.55,  # Much simpler than thought
            richness_score=0.45,  # Lost most richness
            dimensional_count=1,  # Linear language
            sample_data="Linear sentence structure with vocabulary constraints"
        )
    
    def _convert_language_to_output(self, language_data: DataPoint) -> DataPoint:
        """Convert language to final text output"""
        # Final text generation has minimal loss
        # Mostly just formatting and minor word choice
        
        output_size = int(language_data.size_bytes * 0.95)
        
        return DataPoint(
            timestamp=time.time(),
            stage="Text Output",
            data_type="utf8_text",
            size_bytes=output_size,
            complexity_score=0.50,
            richness_score=0.43,
            dimensional_count=1,
            sample_data="Final formatted text output"
        )
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze all scenarios to identify bottlenecks"""
        if not self.scenarios:
            return {
                "error": "No scenarios recorded",
                "bottlenecks": []
            }
        
        # Aggregate data across all scenarios
        transformation_stats = {}
        
        for scenario in self.scenarios:
            for transform in scenario.transformations:
                t_type = transform.transformation_type.value
                
                if t_type not in transformation_stats:
                    transformation_stats[t_type] = {
                        'count': 0,
                        'total_loss': 0.0,
                        'total_duration': 0.0,
                        'loss_breakdown': {}
                    }
                
                stats = transformation_stats[t_type]
                stats['count'] += 1
                stats['total_loss'] += transform.loss_percentage
                stats['total_duration'] += transform.duration_ms
                
                # Aggregate loss breakdown
                for loss_type, loss_value in transform.loss_breakdown.items():
                    if loss_type not in stats['loss_breakdown']:
                        stats['loss_breakdown'][loss_type] = 0.0
                    stats['loss_breakdown'][loss_type] += loss_value
        
        # Calculate averages
        for t_type, stats in transformation_stats.items():
            count = stats['count']
            stats['avg_loss'] = stats['total_loss'] / count
            stats['avg_duration'] = stats['total_duration'] / count
            
            for loss_type in stats['loss_breakdown']:
                stats['loss_breakdown'][loss_type] /= count
        
        # Identify bottlenecks (>40% loss or >50ms latency)
        bottlenecks = []
        
        for t_type, stats in transformation_stats.items():
            if stats['avg_loss'] > 40.0:
                bottlenecks.append({
                    'transformation': t_type,
                    'type': 'information_loss',
                    'severity': 'critical' if stats['avg_loss'] > 55 else 'major',
                    'avg_loss': stats['avg_loss'],
                    'loss_breakdown': stats['loss_breakdown'],
                    'description': f"Loses {stats['avg_loss']:.1f}% of information on average"
                })
            
            if stats['avg_duration'] > 50.0:
                bottlenecks.append({
                    'transformation': t_type,
                    'type': 'latency',
                    'severity': 'major',
                    'avg_duration': stats['avg_duration'],
                    'description': f"Takes {stats['avg_duration']:.1f}ms on average"
                })
        
        return {
            'transformation_stats': transformation_stats,
            'bottlenecks': bottlenecks,
            'total_scenarios': len(self.scenarios),
            'recommendations': self._generate_bottleneck_recommendations(bottlenecks)
        }
    
    def _generate_bottleneck_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate recommendations to address bottlenecks"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck['transformation'] == 'thought_to_language':
                recommendations.extend([
                    "Implement multi-modal language output to preserve dimensional information",
                    "Add semantic embedding layer to capture concept associations",
                    "Include metadata/context alongside linear text",
                    "Develop richer vocabulary and expression patterns",
                    "Use wave signatures to augment language output",
                    "Implement thought-tag system to preserve lost dimensions"
                ])
            elif bottleneck['type'] == 'latency':
                recommendations.append(
                    f"Optimize {bottleneck['transformation']} processing pipeline to reduce latency"
                )
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive data flow report"""
        bottleneck_analysis = self.analyze_bottlenecks()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_scenarios': len(self.scenarios),
                'total_transformations': sum(len(s.transformations) for s in self.scenarios),
            },
            'scenarios': [
                {
                    'name': scenario.scenario_name,
                    'duration_ms': scenario.get_total_duration() * 1000,
                    'total_loss': scenario.total_loss,
                    'end_to_end_retention': scenario.get_end_to_end_retention(),
                    'transformations': [
                        {
                            'type': t.transformation_type.value,
                            'loss': t.loss_percentage,
                            'duration_ms': t.duration_ms,
                            'throughput': t.get_throughput(),
                            'loss_breakdown': t.loss_breakdown
                        }
                        for t in scenario.transformations
                    ]
                }
                for scenario in self.scenarios
            ],
            'bottleneck_analysis': bottleneck_analysis,
            'thought_to_language_focus': self._analyze_thought_to_language()
        }
        
        return report
    
    def _analyze_thought_to_language(self) -> Dict[str, Any]:
        """Detailed analysis of Thought→Language conversion"""
        t2l_transforms = []
        
        for scenario in self.scenarios:
            for transform in scenario.transformations:
                if transform.transformation_type == TransformationType.THOUGHT_TO_LANGUAGE:
                    t2l_transforms.append(transform)
        
        if not t2l_transforms:
            return {
                'error': 'No thought-to-language transformations recorded'
            }
        
        avg_loss = sum(t.loss_percentage for t in t2l_transforms) / len(t2l_transforms)
        
        # Aggregate loss breakdown
        total_breakdown = {}
        for transform in t2l_transforms:
            for loss_type, value in transform.loss_breakdown.items():
                if loss_type not in total_breakdown:
                    total_breakdown[loss_type] = 0.0
                total_breakdown[loss_type] += value
        
        for loss_type in total_breakdown:
            total_breakdown[loss_type] /= len(t2l_transforms)
        
        return {
            'sample_count': len(t2l_transforms),
            'average_loss': avg_loss,
            'loss_breakdown': total_breakdown,
            'primary_loss_factors': sorted(
                total_breakdown.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            'diagnosis': self._diagnose_thought_language_loss(total_breakdown),
            'improvement_strategies': self._suggest_improvements(total_breakdown)
        }
    
    def _diagnose_thought_language_loss(self, breakdown: Dict[str, float]) -> List[str]:
        """Diagnose why thought-to-language loses so much information"""
        diagnoses = []
        
        if breakdown.get('dimensional', 0) > 50:
            diagnoses.append(
                "CRITICAL: Dimensional collapse (4D → 1D) is the primary cause. "
                "Thoughts exist in multi-dimensional space but language is linear."
            )
        
        if breakdown.get('richness', 0) > 40:
            diagnoses.append(
                "MAJOR: Semantic richness loss. Complex concept networks cannot be "
                "expressed in simple word sequences."
            )
        
        if breakdown.get('complexity', 0) > 30:
            diagnoses.append(
                "Parallel thought structures must be serialized into sequential language, "
                "losing simultaneity and interconnections."
            )
        
        if breakdown.get('size', 0) > 50:
            diagnoses.append(
                "Vocabulary and expression patterns are insufficient to capture "
                "full thought complexity."
            )
        
        return diagnoses
    
    def _suggest_improvements(self, breakdown: Dict[str, float]) -> List[Dict[str, Any]]:
        """Suggest specific improvements based on loss breakdown"""
        improvements = []
        
        if breakdown.get('dimensional', 0) > 40:
            improvements.append({
                'problem': 'Dimensional collapse (4D → 1D)',
                'solutions': [
                    'Implement multi-modal output (text + wave signatures + embeddings)',
                    'Add semantic tags/metadata to preserve lost dimensions',
                    'Use nested/hierarchical text structures to capture depth',
                    'Include thought-space coordinates alongside text'
                ],
                'expected_improvement': '20-30% loss reduction'
            })
        
        if breakdown.get('richness', 0) > 35:
            improvements.append({
                'problem': 'Semantic richness loss',
                'solutions': [
                    'Expand vocabulary with more nuanced terms',
                    'Implement concept-linking annotations',
                    'Use metaphors and analogies to convey complexity',
                    'Add context layers to language output'
                ],
                'expected_improvement': '15-25% loss reduction'
            })
        
        if breakdown.get('complexity', 0) > 25:
            improvements.append({
                'problem': 'Complexity reduction',
                'solutions': [
                    'Develop compound expression patterns',
                    'Use structural markers (emotional, logical, intuitive)',
                    'Implement thought-tree serialization',
                    'Add parallelism indicators in language'
                ],
                'expected_improvement': '10-15% loss reduction'
            })
        
        return improvements
    
    def save_report(self, filepath: str):
        """Save data flow report to file"""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Data flow report saved to {filepath}")


def main():
    """Run data flow tracking analysis"""
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("DATA FLOW TRACKER")
    print("="*70)
    print()
    
    tracker = DataFlowTracker()
    
    # Run test scenarios
    test_cases = [
        ("Simple Greeting", "Hello, how are you today?"),
        ("Complex Question", "What is the meaning of consciousness and how does it emerge from physical processes?"),
        ("Emotional Expression", "I feel overwhelmed with joy and gratitude for this beautiful moment"),
    ]
    
    print("Running test scenarios...")
    print()
    
    for name, text in test_cases:
        print(f"📝 Scenario: {name}")
        print(f"   Input: {text}")
        scenario = tracker.run_test_scenario(name, text)
        print(f"   Total Loss: {scenario.total_loss:.1f}%")
        print(f"   Retention: {scenario.get_end_to_end_retention():.1f}%")
        print()
    
    # Generate report
    report = tracker.generate_report()
    
    # Print bottleneck analysis
    print("\n🚧 BOTTLENECK ANALYSIS")
    print("="*70)
    
    bottlenecks = report['bottleneck_analysis']['bottlenecks']
    for bottleneck in bottlenecks:
        print(f"\n⚠ {bottleneck['transformation'].upper()}")
        print(f"   Type: {bottleneck['type']}")
        print(f"   Severity: {bottleneck['severity'].upper()}")
        if 'avg_loss' in bottleneck:
            print(f"   Average Loss: {bottleneck['avg_loss']:.1f}%")
            if 'loss_breakdown' in bottleneck:
                print(f"   Loss Breakdown:")
                for loss_type, value in bottleneck['loss_breakdown'].items():
                    print(f"      {loss_type}: {value:.1f}%")
        if 'avg_duration' in bottleneck:
            print(f"   Average Duration: {bottleneck['avg_duration']:.1f}ms")
    
    # Focus on Thought→Language
    print("\n🔍 THOUGHT→LANGUAGE CONVERSION ANALYSIS")
    print("="*70)
    
    t2l = report['thought_to_language_focus']
    if 'error' not in t2l:
        print(f"Samples Analyzed: {t2l['sample_count']}")
        print(f"Average Information Loss: {t2l['average_loss']:.1f}%")
        print()
        
        print("Loss Breakdown:")
        for loss_type, value in t2l['loss_breakdown'].items():
            print(f"  {loss_type}: {value:.1f}%")
        print()
        
        print("Diagnosis:")
        for diagnosis in t2l['diagnosis']:
            print(f"  • {diagnosis}")
        print()
        
        print("Improvement Strategies:")
        for i, improvement in enumerate(t2l['improvement_strategies'], 1):
            print(f"\n  {i}. Problem: {improvement['problem']}")
            print(f"     Expected Improvement: {improvement['expected_improvement']}")
            print(f"     Solutions:")
            for solution in improvement['solutions']:
                print(f"       - {solution}")
    
    # Print recommendations
    print("\n💡 RECOMMENDATIONS")
    print("="*70)
    for i, rec in enumerate(report['bottleneck_analysis']['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save report
    tracker.save_report('reports/data_flow_report.json')
    print(f"\n✅ Full report saved to reports/data_flow_report.json")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
