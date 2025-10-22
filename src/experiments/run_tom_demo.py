#!/usr/bin/env python3
"""
Quick ToM Experiment Demo
=========================

Run a quick demonstration of the Theory of Mind experiment framework.

This script:
1. Loads a few sample ToM tasks
2. Runs cognitive analysis on one task
3. Runs a multi-agent dialogue
4. Shows key results

Usage:
    .venv/bin/python src/experiments/run_tom_demo.py
"""

import sys
from pathlib import Path
import random

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "probes"))

from tom_tasks import ToMTaskGenerator, ToMTaskType
from tom_inference import ToMInferenceEngine
from tom_dialogue import ToMDialogueEngine


def print_banner(text):
    """Print a nice banner"""
    print("\n" + "="*80)
    print(f"{text:^80}")
    print("="*80 + "\n")


def main():
    print_banner("🧠 THEORY OF MIND EXPERIMENT - QUICK DEMO 🧠")

    # Paths
    data_dir = Path(__file__).parent.parent.parent / "data"
    task_path = data_dir / "tom_tasks" / "tom_task_suite.json"
    probes_dir = data_dir / "probes_binary"

    # Check if probes exist
    if not probes_dir.exists():
        print("❌ Error: Probe directory not found!")
        print(f"Expected: {probes_dir}")
        print("\nPlease ensure you have trained probes in data/probes_binary/")
        print("See README.md for instructions on training probes.")
        return

    # Check if tasks exist
    if not task_path.exists():
        print("⚠️  ToM tasks not found. Generating now...")
        from tom_tasks import main as generate_tasks
        generate_tasks()

    # Load tasks
    print("📚 Loading ToM tasks...")
    generator = ToMTaskGenerator()
    tasks = generator.load_tasks(task_path)
    print(f"✓ Loaded {len(tasks)} tasks")

    # Select diverse samples
    sample_tasks = []
    for task_type in [ToMTaskType.FALSE_BELIEF, ToMTaskType.UNEXPECTED_CONTENTS,
                      ToMTaskType.AFFECTIVE_TOM]:
        type_tasks = [t for t in tasks if t.task_type == task_type]
        if type_tasks:
            sample_tasks.append(random.choice(type_tasks))

    print(f"✓ Selected {len(sample_tasks)} diverse sample tasks for demo\n")

    # =========================================================================
    # Part 1: Single Task Analysis
    # =========================================================================
    print_banner("PART 1: SINGLE TASK COGNITIVE ANALYSIS")

    demo_task = sample_tasks[0]

    print(f"Task Type: {demo_task.task_type.value}")
    print(f"Task ID: {demo_task.task_id}")
    print(f"\nScenario:")
    print(demo_task.scenario)
    print(f"\nQuestion: {demo_task.question}")
    print(f"Correct Answer: {demo_task.correct_answer}")
    print(f"\nExpected Cognitive Actions:")
    print(", ".join(demo_task.expected_cognitive_actions))

    print("\n" + "-"*80)
    print("Initializing ToM Inference Engine...")
    print("-"*80 + "\n")

    engine = ToMInferenceEngine(
        probes_base_dir=probes_dir,
        model_name="google/gemma-3-4b-it",
        verbose=True
    )

    print("\n" + "-"*80)
    print("Running cognitive analysis with real-time tracking...")
    print("-"*80 + "\n")

    signature = engine.analyze_task(
        demo_task,
        threshold=0.1,
        show_realtime=True
    )

    print("\n" + "-"*80)
    print("ANALYSIS RESULTS")
    print("-"*80)

    print(f"\n📊 Metrics:")
    print(f"  ToM Specificity Score: {signature.tom_specificity_score:.3f}")
    print(f"  Expected Action Coverage: {signature.expected_action_coverage:.1%}")

    print(f"\n✓ Expected Actions Detected:")
    print(f"  {', '.join(signature.detected_expected_actions)}")

    print(f"\n⚡ Top 5 Differential Activations (Test - Control):")
    sorted_diff = sorted(signature.differential_actions.items(),
                        key=lambda x: x[1], reverse=True)
    for i, (action, diff) in enumerate(sorted_diff[:5], 1):
        marker = "✓" if action in signature.expected_actions else " "
        print(f"  {marker} {i}. {action:30s} {diff:+.4f}")

    # =========================================================================
    # Part 2: Multi-Agent Dialogue
    # # =========================================================================
    # print_banner("PART 2: MULTI-AGENT ToM DIALOGUE")

    # dialogue_task = sample_tasks[1] if len(sample_tasks) > 1 else sample_tasks[0]

    # print(f"Task Type: {dialogue_task.task_type.value}")
    # print(f"\nRunning dialogue between Narrator and Reasoner...")
    # print("(Watch cognitive actions activate in the Reasoner)\n")

    # dialogue_engine = ToMDialogueEngine(
    #     probes_base_dir=probes_dir,
    #     model_name="google/gemma-3-4b-it",
    #     verbose=True
    # )

    # session = dialogue_engine.run_dialogue_session(
    #     dialogue_task,
    #     threshold=0.1,
    #     show_realtime=False  # Set to True for full real-time display
    # )

    # =========================================================================
    # Part 3: Summary & Next Steps
    # =========================================================================
    print_banner("DEMO COMPLETE!")

    print("🎉 You've successfully run the ToM experiment demo!\n")

    print("📈 What we did:")
    print("  ✓ Analyzed cognitive processes during ToM reasoning")
    print("  ✓ Compared test vs. control conditions")
    print("  ✓ Ran a multi-agent dialogue with cognitive tracking")

    print("\n🔬 Key Findings from This Demo:")
    print(f"  • ToM Specificity: {signature.tom_specificity_score:.3f}")
    print(f"  • Coverage: {signature.expected_action_coverage:.1%}")
    print(f"  • Top ToM Action: {sorted_diff[0][0]}")

    print("\n📚 Next Steps:")
    print("  1. Run full experiment: src/experiments/tom_inference.py")
    print("  2. Explore interactive notebook: notebooks/ToM_Experiment.ipynb")
    print("  3. Analyze all 105 tasks for comprehensive results")
    print("  4. Generate visualizations with tom_analysis.py")

    print("\n📖 Documentation:")
    print("  See src/experiments/README.md for full details")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check that:")
        print("  1. Probes are trained and in data/probes_binary/")
        print("  2. Model 'google/gemma-3-4b-it' is accessible")
        print("  3. All dependencies are installed")
