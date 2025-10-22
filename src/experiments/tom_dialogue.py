"""
Multi-Agent Theory of Mind Dialogue System
==========================================

Two AI agents engage in dialogues about ToM scenarios while we track
their cognitive processes in real-time.

Agent Roles:
- Narrator: Presents ToM scenarios and asks questions
- Reasoner: Answers ToM questions (their cognition is tracked)

This extends the multi-agent therapy conversation framework for ToM research.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import random

# Add necessary paths
PROBES_PATH = Path(__file__).parent.parent / "probes"
sys.path.insert(0, str(PROBES_PATH))

from streaming_probe_inference import StreamingProbeInferenceEngine, AggregatedPrediction
from tom_tasks import ToMTask, ToMTaskType
from tom_inference import ToMCognitiveSignature


@dataclass
class ToMDialogueTurn:
    """Single turn in ToM dialogue"""
    turn_number: int
    speaker: str  # "narrator" or "reasoner"
    content: str
    cognitive_predictions: Optional[List[AggregatedPrediction]] = None
    tom_actions_detected: Optional[List[str]] = None


@dataclass
class ToMDialogueSession:
    """Complete ToM dialogue session"""
    session_id: str
    task: ToMTask
    turns: List[ToMDialogueTurn]

    # Summary statistics
    reasoner_tom_actions_count: Dict[str, int]
    critical_reasoning_turns: List[int]  # Turns with high ToM action density


class ToMDialogueEngine:
    """
    Multi-agent dialogue engine for ToM experiments

    Creates conversations where:
    1. Narrator presents a ToM scenario
    2. Reasoner thinks aloud about the scenario
    3. Narrator asks the ToM question
    4. Reasoner provides answer with reasoning
    """

    def __init__(
        self,
        probes_base_dir: Path,
        model_name: str = "google/gemma-3-4b-it",
        device: str = None,
        verbose: bool = True
    ):
        """Initialize ToM dialogue engine"""
        self.verbose = verbose

        # Initialize inference engine
        self.engine = StreamingProbeInferenceEngine(
            probes_base_dir=probes_base_dir,
            model_name=model_name,
            device=device,
            verbose=verbose,
            layer_range=(21, 30)
        )

        # ToM-specific actions to track
        self.tom_actions = [
            "perspective_taking",
            "hypothesis_generation",
            "metacognitive_monitoring",
            "distinguishing",
            "updating_beliefs",
            "counterfactual_reasoning"
        ]

        # System prompts for agents
        self.narrator_prompt = (
            "You are a narrator presenting theory of mind scenarios. "
            "Present the scenario clearly and ask the key question. "
            "Be concise and neutral."
        )

        self.reasoner_prompt = (
            "You are reasoning about theory of mind scenarios. "
            "Think step-by-step about what different people know, believe, and feel. "
            "Explain your reasoning clearly, distinguishing between:\n"
            "- What actually happened (reality)\n"
            "- What different characters believe (their mental states)\n"
            "- What characters think other characters believe (nested beliefs)\n"
            "Take your time to reason carefully before answering."
        )

    def run_dialogue_session(
        self,
        task: ToMTask,
        threshold: float = 0.1,
        show_realtime: bool = True
    ) -> ToMDialogueSession:
        """
        Run a complete ToM dialogue session

        Args:
            task: ToMTask to discuss
            threshold: Confidence threshold for cognitive actions
            show_realtime: Show real-time cognitive activations

        Returns:
            ToMDialogueSession with full conversation and analysis
        """
        session_id = f"tom_dialogue_{task.task_id}"
        turns = []

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ToM DIALOGUE SESSION: {session_id}")
            print(f"Task Type: {task.task_type.value}")
            print(f"{'='*80}\n")

        # Turn 1: Narrator presents scenario
        turn1_content = f"Let me present a scenario:\n\n{task.scenario}"
        turns.append(ToMDialogueTurn(
            turn_number=1,
            speaker="narrator",
            content=turn1_content
        ))

        if self.verbose:
            print(f"NARRATOR (Turn 1):\n{turn1_content}\n")
            print("-" * 80 + "\n")

        # Turn 2: Reasoner thinks aloud about the scenario
        turn2_prompt = (
            f"{self.reasoner_prompt}\n\n"
            f"Scenario: {task.scenario}\n\n"
            "Think about this scenario step by step. What are the key facts? "
            "What does each person know or believe? Provide your reasoning:"
        )

        if self.verbose:
            print(f"REASONER (Turn 2) - Initial Reasoning:")

        # Generate response while tracking cognitive actions
        turn2_response, turn2_preds_raw = self.engine.generate_with_cognitive_tracking(
            turn2_prompt,
            max_new_tokens=256,
            threshold=0.0,  # Get all predictions for proper aggregation
            show_realtime=show_realtime,
            temperature=0.7
        )
        # Aggregate predictions by action across layers (like Interactive_TUI.py)
        turn2_agg = self.engine.aggregate_predictions(turn2_preds_raw, threshold=threshold)
        turn2_tom_actions = [pred.action_name for pred in turn2_agg
                             if pred.is_active and pred.action_name in self.tom_actions]

        turn2_content = turn2_response
        turns.append(ToMDialogueTurn(
            turn_number=2,
            speaker="reasoner",
            content=turn2_content,
            cognitive_predictions=turn2_agg,
            tom_actions_detected=turn2_tom_actions
        ))

        if self.verbose:
            print(f"\n{turn2_content}\n")
            print(f"ToM Actions Detected: {', '.join(turn2_tom_actions)}")
            print("-" * 80 + "\n")

        # Turn 3: Narrator asks the question
        turn3_content = f"Question: {task.question}"
        turns.append(ToMDialogueTurn(
            turn_number=3,
            speaker="narrator",
            content=turn3_content
        ))

        if self.verbose:
            print(f"NARRATOR (Turn 3):\n{turn3_content}\n")
            print("-" * 80 + "\n")

        # Turn 4: Reasoner answers with full reasoning
        turn4_prompt = (
            f"{self.reasoner_prompt}\n\n"
            f"Scenario: {task.scenario}\n\n"
            f"Question: {task.question}\n\n"
            "Provide your answer and explain your reasoning step by step. "
            "Make sure to distinguish between what actually happened and what "
            "each person believes or knows. Answer:"
        )

        if self.verbose:
            print(f"REASONER (Turn 4) - Final Answer & Reasoning:")

        # Generate response while tracking cognitive actions
        turn4_response, turn4_preds_raw = self.engine.generate_with_cognitive_tracking(
            turn4_prompt,
            max_new_tokens=256,
            threshold=0.0,  # Get all predictions for proper aggregation
            show_realtime=show_realtime,
            temperature=0.7
        )
        # Aggregate predictions by action across layers (like Interactive_TUI.py)
        turn4_agg = self.engine.aggregate_predictions(turn4_preds_raw, threshold=threshold)
        turn4_tom_actions = [pred.action_name for pred in turn4_agg
                             if pred.is_active and pred.action_name in self.tom_actions]

        turn4_content = turn4_response
        turns.append(ToMDialogueTurn(
            turn_number=4,
            speaker="reasoner",
            content=turn4_content,
            cognitive_predictions=turn4_agg,
            tom_actions_detected=turn4_tom_actions
        ))

        if self.verbose:
            print(f"\n{turn4_content}\n")
            print(f"ToM Actions Detected: {', '.join(turn4_tom_actions)}")
            print("-" * 80 + "\n")

        # Aggregate statistics
        tom_action_counts = {}
        for turn in turns:
            if turn.tom_actions_detected:
                for action in turn.tom_actions_detected:
                    tom_action_counts[action] = tom_action_counts.get(action, 0) + 1

        # Find critical reasoning turns (high ToM action density)
        critical_turns = [
            turn.turn_number for turn in turns
            if turn.tom_actions_detected and len(turn.tom_actions_detected) >= 3
        ]

        session = ToMDialogueSession(
            session_id=session_id,
            task=task,
            turns=turns,
            reasoner_tom_actions_count=tom_action_counts,
            critical_reasoning_turns=critical_turns
        )

        if self.verbose:
            self._print_session_summary(session)

        return session

    def _generate_reasoning(self, task: ToMTask, stage: str) -> str:
        """Generate template reasoning based on task type"""
        if task.task_type == ToMTaskType.FALSE_BELIEF:
            if stage == "initial":
                return (
                    f"I notice that {task.characters[0]} placed the {task.objects[0]} in one location, "
                    f"then left. While they were gone, {task.characters[1]} moved it. "
                    f"This means {task.characters[0]} doesn't know about the move."
                )
            else:
                return (
                    f"{task.characters[0]} will look where they last saw the {task.objects[0]}, "
                    f"because they don't know it was moved. Their belief is outdated - they think "
                    f"it's still in {task.locations[0]}, even though it's actually in {task.locations[1]}."
                )

        elif task.task_type == ToMTaskType.UNEXPECTED_CONTENTS:
            if stage == "initial":
                return (
                    f"The container looks like it should have one thing inside, but actually has something else. "
                    f"The key is that {task.characters[0]} hasn't looked inside yet."
                )
            else:
                return (
                    f"{task.characters[0]} will think the container has what it normally contains, "
                    f"because they haven't looked inside. They don't have the privileged knowledge "
                    f"that we have about the actual contents."
                )

        elif task.task_type == ToMTaskType.APPEARANCE_REALITY:
            if stage == "initial":
                return (
                    f"The object appears to be one thing but is actually another. "
                    f"{task.characters[0]} can only see it from far away, so they see the appearance, not the reality."
                )
            else:
                return (
                    f"{task.characters[0]} will think it's a {task.objects[1]} because that's what it looks like. "
                    f"They haven't examined it closely enough to discover it's actually a {task.objects[0]}."
                )

        elif task.task_type == ToMTaskType.SECOND_ORDER_BELIEF:
            if stage == "initial":
                return (
                    f"This is complex - I need to track what {task.characters[0]} thinks that "
                    f"{task.characters[1]} thinks. {task.characters[0]} knows the {task.objects[0]} moved, "
                    f"but doesn't know that {task.characters[1]} doesn't know."
                )
            else:
                return (
                    f"{task.characters[0]} thinks {task.characters[1]} will look in the original location, "
                    f"because {task.characters[0]} doesn't realize that {task.characters[1]} wasn't informed "
                    f"about the move. This requires reasoning about nested beliefs."
                )

        else:  # Affective ToM
            if stage == "initial":
                return (
                    f"I need to consider how this situation would make {task.characters[0]} feel "
                    f"based on what happened to them."
                )
            else:
                return (
                    f"Based on the situation, {task.characters[0]} would likely feel this way because "
                    f"of how events unfolded and what they experienced."
                )

    def _print_session_summary(self, session: ToMDialogueSession):
        """Print session summary"""
        print(f"\n{'='*80}")
        print(f"SESSION SUMMARY")
        print(f"{'='*80}")

        print(f"\nTask: {session.task.task_type.value} ({session.task.task_id})")
        print(f"Total Turns: {len(session.turns)}")
        print(f"Critical Reasoning Turns: {session.critical_reasoning_turns}")

        print(f"\nToM Actions Detected in Reasoner:")
        for action, count in sorted(session.reasoner_tom_actions_count.items(),
                                    key=lambda x: x[1], reverse=True):
            print(f"  {action}: {count} turn(s)")

    def run_multi_session_experiment(
        self,
        tasks: List[ToMTask],
        threshold: float = 0.1,
        save_path: Optional[Path] = None
    ) -> List[ToMDialogueSession]:
        """
        Run dialogue sessions for multiple tasks

        Args:
            tasks: List of ToMTask objects
            threshold: Confidence threshold
            save_path: Optional path to save results

        Returns:
            List of ToMDialogueSession objects
        """
        print(f"\n{'='*80}")
        print(f"MULTI-SESSION ToM DIALOGUE EXPERIMENT")
        print(f"{'='*80}")
        print(f"Running {len(tasks)} dialogue sessions...\n")

        sessions = []
        for i, task in enumerate(tasks, 1):
            print(f"\n[Session {i}/{len(tasks)}]")
            session = self.run_dialogue_session(
                task,
                threshold=threshold,
                show_realtime=False  # Don't show realtime for bulk runs
            )
            sessions.append(session)

        # Save if requested
        if save_path:
            self._save_sessions(sessions, save_path)

        # Print aggregate summary
        self._print_aggregate_summary(sessions)

        return sessions

    def _save_sessions(self, sessions: List[ToMDialogueSession], save_path: Path):
        """Save dialogue sessions to JSON"""
        sessions_data = []
        for session in sessions:
            session_dict = {
                'session_id': session.session_id,
                'task': {
                    'task_id': session.task.task_id,
                    'task_type': session.task.task_type.value,
                    'scenario': session.task.scenario,
                    'question': session.task.question,
                    'correct_answer': session.task.correct_answer
                },
                'turns': [
                    {
                        'turn_number': turn.turn_number,
                        'speaker': turn.speaker,
                        'content': turn.content,
                        'tom_actions_detected': turn.tom_actions_detected
                    }
                    for turn in session.turns
                ],
                'reasoner_tom_actions_count': session.reasoner_tom_actions_count,
                'critical_reasoning_turns': session.critical_reasoning_turns
            }
            sessions_data.append(session_dict)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(sessions_data, f, indent=2)

        print(f"\n✓ Saved {len(sessions)} dialogue sessions to {save_path}")

    def _print_aggregate_summary(self, sessions: List[ToMDialogueSession]):
        """Print summary across all sessions"""
        print(f"\n{'='*80}")
        print(f"AGGREGATE SUMMARY ({len(sessions)} sessions)")
        print(f"{'='*80}")

        # Count ToM actions across all sessions
        all_tom_actions = {}
        for session in sessions:
            for action, count in session.reasoner_tom_actions_count.items():
                all_tom_actions[action] = all_tom_actions.get(action, 0) + count

        print(f"\nMost Frequent ToM Actions Across All Dialogues:")
        for action, count in sorted(all_tom_actions.items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {action}: {count} occurrences")

        # Average critical turns per session
        avg_critical = sum(len(s.critical_reasoning_turns) for s in sessions) / len(sessions)
        print(f"\nAverage Critical Reasoning Turns per Session: {avg_critical:.1f}")


def main():
    """Example usage"""
    from tom_tasks import ToMTaskGenerator

    # Initialize
    engine = ToMDialogueEngine(
        probes_base_dir=Path("data/probes_binary"),
        model_name="google/gemma-3-4b-it",
        verbose=True
    )

    # Load tasks
    task_path = Path("data/tom_tasks/tom_task_suite.json")
    if task_path.exists():
        generator = ToMTaskGenerator()
        tasks = generator.load_tasks(task_path)

        # Run dialogue on first few tasks
        sample_tasks = random.sample(tasks, min(3, len(tasks)))

        sessions = engine.run_multi_session_experiment(
            sample_tasks,
            threshold=0.1,
            save_path=Path("output/tom_experiments/dialogue_sessions.json")
        )
    else:
        print(f"Task suite not found at {task_path}")


if __name__ == "__main__":
    main()
