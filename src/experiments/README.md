# Theory of Mind (ToM) Cognitive Action Experiments

A comprehensive experimental framework for studying Theory of Mind reasoning in language models using real-time cognitive action detection.

## Overview

Instead of just measuring whether models answer ToM questions correctly, this framework reveals **which cognitive processes activate** during mental state attribution, creating a "cognitive fingerprint" of Theory of Mind reasoning.

## What Makes This Unique

- **Real-Time Tracking**: Monitor 45 cognitive actions token-by-token during ToM reasoning
- **Comparative Analysis**: Control vs. test conditions isolate ToM-specific cognition
- **Multi-Agent Dialogues**: Watch cognitive patterns emerge during AI-to-AI ToM discussions
- **Classic Tasks**: 105 validated tasks from developmental psychology (Sally-Anne, Smarties, etc.)
- **Layer Analysis**: Identify which model layers are responsible for ToM reasoning

## Quick Start

### 1. Generate ToM Tasks

```bash
.venv/bin/python src/experiments/tom_tasks.py
```

This creates 105 ToM tasks across 5 categories:
- **False Belief** (30 tasks): Sally-Anne paradigm
- **Unexpected Contents** (20 tasks): Smarties paradigm
- **Appearance-Reality** (20 tasks): Dual representation
- **Second-Order Belief** (15 tasks): Recursive mental states
- **Affective ToM** (20 tasks): Emotion inference

### 2. Run Single Task Analysis

```python
from pathlib import Path
from tom_tasks import ToMTaskGenerator
from tom_inference import ToMInferenceEngine

# Initialize
engine = ToMInferenceEngine(
    probes_base_dir=Path("data/probes_binary"),
    model_name="google/gemma-3-4b-it"
)

# Load tasks
generator = ToMTaskGenerator()
tasks = generator.load_tasks(Path("data/tom_tasks/tom_task_suite.json"))

# Analyze a task
signature = engine.analyze_task(tasks[0], threshold=0.1, show_realtime=True)

print(f"ToM Specificity: {signature.tom_specificity_score:.3f}")
print(f"Expected Coverage: {signature.expected_action_coverage:.1%}")
```

### 3. Run Multi-Agent Dialogue

```python
from tom_dialogue import ToMDialogueEngine

dialogue_engine = ToMDialogueEngine(
    probes_base_dir=Path("data/probes_binary"),
    model_name="google/gemma-3-4b-it"
)

session = dialogue_engine.run_dialogue_session(
    tasks[0],
    threshold=0.1,
    show_realtime=True
)
```

### 4. Batch Analysis & Visualization

```python
from tom_analysis import ToMAnalyzer

# Analyze multiple tasks
result = engine.analyze_task_suite(
    tasks[:20],
    save_path=Path("output/tom_experiments/results.json")
)

# Generate visualizations
analyzer = ToMAnalyzer(output_dir=Path("output/tom_experiments/viz"))
analyzer.create_comprehensive_report(result, tasks[:20])
```

### 5. Interactive Notebook

```bash
jupyter notebook notebooks/ToM_Experiment.ipynb
```

## Experiment Components

### `tom_tasks.py` - Task Generation System

**Classes:**
- `ToMTask`: Single ToM task with scenario, question, control condition
- `ToMTaskGenerator`: Generate and manage task suites

**Task Types:**
```python
class ToMTaskType(Enum):
    FALSE_BELIEF = "false_belief"
    UNEXPECTED_CONTENTS = "unexpected_contents"
    APPEARANCE_REALITY = "appearance_reality"
    SECOND_ORDER_BELIEF = "second_order_belief"
    AFFECTIVE_TOM = "affective_tom"
```

**Example Task:**
```python
ToMTask(
    task_id="false_belief_0001",
    scenario="Emma puts a book in the box and leaves. Sally moves it to the drawer.",
    question="Where will Emma look for the book?",
    correct_answer="in the box",
    expected_cognitive_actions=["perspective_taking", "distinguishing", ...]
)
```

### `tom_inference.py` - Cognitive Analysis Engine

**Classes:**
- `ToMInferenceEngine`: Extends `StreamingProbeInferenceEngine` for ToM analysis
- `ToMCognitiveSignature`: Single task cognitive profile
- `ToMExperimentResult`: Aggregate results across tasks

**Key Metrics:**
- **ToM Specificity Score**: How much more ToM actions activate vs. control
- **Expected Action Coverage**: % of expected cognitive actions detected
- **Differential Activations**: Test confidence - Control confidence for each action
- **Layer Preferences**: Which layers show strongest ToM signals

**Analysis Pipeline:**
```
Task → Test Scenario → Probe Inference → Cognitive Actions (Test)
    → Control Scenario → Probe Inference → Cognitive Actions (Control)
    → Differential Analysis → ToM Signature
```

### `tom_dialogue.py` - Multi-Agent System

**Dialogue Structure:**
1. **Narrator**: Presents ToM scenario
2. **Reasoner**: Thinks aloud (cognitive actions tracked)
3. **Narrator**: Asks ToM question
4. **Reasoner**: Provides answer with reasoning (tracked)

**Tracked Metrics:**
- ToM action counts per turn
- Critical reasoning turns (high ToM action density)
- Temporal dynamics of cognitive processes

### `tom_analysis.py` - Visualization Tools

**Generated Visualizations:**
1. **Heatmap**: Cognitive actions × Task types
2. **Differential Bar Chart**: ToM-specific actions ranked
3. **Layer Profile**: Activation by layer (21-30)
4. **Specificity Distribution**: Histogram + boxplots
5. **Coverage Analysis**: Expected action detection rates
6. **Token Timelines**: Activation patterns over tokens
7. **Co-occurrence Network**: Which actions co-activate
8. **Summary Report**: Text-based statistics

## Expected Cognitive Signatures

### Hypothesis

True ToM reasoning should show:

**High Activation:**
- `perspective_taking` - Represent another's viewpoint
- `hypothesis_generation` - Infer mental states
- `metacognitive_monitoring` - Track own vs. other's knowledge
- `distinguishing` - Separate self/other beliefs
- `counterfactual_reasoning` - Consider alternative scenarios

**Medium Activation:**
- `updating_beliefs` - Revise mental models
- `emotion_perception` - For affective ToM
- `suspending_judgment` - Hold multiple representations

**Layer Pattern:**
- Early layers (21-23): `noticing`, `analyzing`
- Middle layers (24-27): `perspective_taking`, `distinguishing`
- Late layers (28-30): `metacognitive_monitoring`, `hypothesis_generation`

## Example Results

### Sample Task Analysis

```
FALSE BELIEF: Sally-Anne Task
ToM Specificity Score: 0.245
Expected Action Coverage: 80.0%

Top Differential Activations (Test - Control):
  ✓ 1. perspective_taking        +0.3821
  ✓ 2. distinguishing            +0.2947
  ✓ 3. metacognitive_monitoring  +0.2103
  ✓ 4. hypothesis_generation     +0.1856
    5. noticing                  +0.1432

Critical Tokens:
  Token  45: 'while'   → perspective_taking, distinguishing
  Token  67: 'moved'   → updating_beliefs, hypothesis_generation
  Token  89: 'will'    → counterfactual_reasoning, perspective_taking
```

### Layer Preferences

```
Layer 21: ███████░░░░░░░░ 0.1234
Layer 22: ████████░░░░░░░ 0.1567
Layer 23: ██████████░░░░░ 0.2103
Layer 24: ███████████░░░░ 0.2456
Layer 25: ████████████░░░ 0.2789
Layer 26: █████████████░░ 0.2934  ← Peak for perspective_taking
Layer 27: ████████████░░░ 0.2801
Layer 28: ███████████░░░░ 0.2567
Layer 29: ██████████░░░░░ 0.2234
Layer 30: ████████░░░░░░░ 0.1876
```

## Research Questions

This framework enables investigation of:

1. **Mechanistic Understanding**
   - Which cognitive processes constitute ToM in LLMs?
   - Do models use recursive reasoning or shallow heuristics?

2. **Developmental Patterns**
   - How do ToM signatures change with model size?
   - Which cognitive actions emerge first during training?

3. **Task Difficulty**
   - Why are second-order beliefs harder?
   - Do false beliefs require different cognition than affective ToM?

4. **Intervention Studies**
   - What happens if we ablate layers with high `perspective_taking`?
   - Can we enhance ToM by amplifying specific cognitive actions?

5. **Transfer Learning**
   - Do ToM signatures trained on one task type transfer to others?
   - Can we use cognitive fingerprints to predict ToM accuracy?

## Output Files

```
output/tom_experiments/
├── tom_task_suite.json              # All 105 tasks
├── batch_results.json               # Experiment results
├── dialogue_sessions.json           # Multi-agent dialogues
└── visualizations/
    ├── 01_action_by_tasktype_heatmap.png
    ├── 02_differential_activations.png
    ├── 03_layer_preferences.png
    ├── 04_specificity_distribution.png
    ├── 05_expected_coverage.png
    ├── 06_token_timelines.png
    ├── 07_action_network.png
    └── 08_summary_report.txt
```

## Advanced Usage

### Custom Task Creation

```python
from tom_tasks import ToMTask, ToMTaskType, TaskDifficulty

custom_task = ToMTask(
    task_id="custom_001",
    task_type=ToMTaskType.FALSE_BELIEF,
    difficulty=TaskDifficulty.HARD,
    scenario="Your custom scenario...",
    control_scenario="Factual version...",
    question="Your question?",
    correct_answer="Answer",
    alternatives=["Alt 1", "Alt 2"],
    characters=["Alice", "Bob"],
    objects=["key"],
    locations=["box", "drawer"],
    expected_cognitive_actions=["perspective_taking", "distinguishing"],
    tom_explanation="Why this requires ToM..."
)
```

### Filter Tasks by Criteria

```python
# Get only hard tasks
hard_tasks = [t for t in tasks if t.difficulty == TaskDifficulty.HARD]

# Get second-order belief tasks
recursive_tasks = [t for t in tasks if t.task_type == ToMTaskType.SECOND_ORDER_BELIEF]

# Get tasks with specific characters
emma_tasks = [t for t in tasks if "Emma" in t.characters]
```

### Export Results for Analysis

```python
import pandas as pd

# Convert results to DataFrame
data = []
for sig in result.task_results:
    data.append({
        'task_id': sig.task_id,
        'task_type': sig.task_type.value,
        'specificity': sig.tom_specificity_score,
        'coverage': sig.expected_action_coverage,
        'n_expected_detected': len(sig.detected_expected_actions),
        'n_unexpected': len(sig.unexpected_actions)
    })

df = pd.DataFrame(data)
df.to_csv('output/tom_experiments/results.csv', index=False)
```

## Dependencies

Required packages (already in Brije environment):
- `torch` - Neural network operations
- `transformers` - Language models
- `nnsight` - Activation extraction
- `matplotlib` - Visualizations
- `seaborn` - Statistical plots
- `numpy` - Numerical operations
- `networkx` - Network graphs (optional)

## Performance Notes

- **Single task analysis**: ~30-60 seconds (2x inference: test + control)
- **10 task batch**: ~5-10 minutes
- **Full 105 task suite**: ~60-90 minutes
- **Memory**: ~8-16GB VRAM for Gemma 3 4B

## Citation

If you use this ToM experiment framework in your research:

```bibtex
@software{brije_tom_experiments,
  title = {Brije: Theory of Mind Cognitive Action Experiments},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/brije}
}
```

Classic ToM papers:
- Wimmer & Perner (1983) - False belief tasks
- Baron-Cohen et al. (1985) - Sally-Anne test
- Perner & Wimmer (1985) - Second-order false beliefs
- Gopnik & Astington (1988) - Unexpected contents

## Future Directions

- [ ] Add SmartGPT-style multi-step reasoning tasks
- [ ] Implement intervention studies (layer ablation)
- [ ] Cross-model comparison framework
- [ ] Temporal dynamics analysis (how signatures evolve during generation)
- [ ] Integration with behavioral accuracy metrics
- [ ] Adversarial ToM tasks (misleading information)
- [ ] Cultural variation in ToM scenarios

## License

Same as Brije main repository.
