"""
Theory of Mind (ToM) Task Suite
================================

Generates classic ToM tasks from developmental psychology to test
AI models' ability to attribute mental states to others.

Task Types:
1. False Belief (Sally-Anne paradigm)
2. Unexpected Contents (Smarties paradigm)
3. Appearance-Reality
4. Second-Order False Belief
5. Affective ToM (emotion/motivation inference)

Each task includes control and test conditions to isolate ToM-specific cognition.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ToMTaskType(Enum):
    """Types of Theory of Mind tasks"""
    FALSE_BELIEF = "false_belief"
    UNEXPECTED_CONTENTS = "unexpected_contents"
    APPEARANCE_REALITY = "appearance_reality"
    SECOND_ORDER_BELIEF = "second_order_belief"
    AFFECTIVE_TOM = "affective_tom"


class TaskDifficulty(Enum):
    """Task difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ToMTask:
    """Single Theory of Mind task"""
    task_id: str
    task_type: ToMTaskType
    difficulty: TaskDifficulty

    # The narrative/scenario
    scenario: str

    # Control text (same scenario without mental state reasoning)
    control_scenario: str

    # The question requiring ToM
    question: str

    # Correct answer
    correct_answer: str

    # Alternative answers (for multiple choice)
    alternatives: List[str]

    # Metadata
    characters: List[str]
    objects: List[str]
    locations: List[str]

    # Expected cognitive actions for ToM reasoning
    expected_cognitive_actions: List[str]

    # Explanation of why this requires ToM
    tom_explanation: str


class ToMTaskGenerator:
    """Generates Theory of Mind tasks with variations"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.task_count = 0

        # Character pools
        self.characters = [
            "Sally", "Anne", "Tom", "Emma", "Jack", "Sophia", "Oliver", "Ava",
            "Liam", "Mia", "Noah", "Isabella", "Ethan", "Charlotte", "Lucas", "Amelia"
        ]

        # Object pools
        self.containers = ["basket", "box", "drawer", "bag", "cupboard", "trunk", "cabinet"]
        self.locations = ["kitchen", "living room", "bedroom", "garden", "playground", "classroom"]
        self.items = ["marble", "toy car", "doll", "ball", "book", "crayon", "cookie", "apple"]

        # For unexpected contents
        self.container_items = {
            "candy box": ["candies", "pencils", "buttons", "coins"],
            "cereal box": ["cereal", "toys", "letters", "photos"],
            "cookie jar": ["cookies", "marbles", "keys", "stamps"],
            "pencil case": ["pencils", "candies", "erasers", "clips"]
        }

        # For appearance-reality
        self.appearance_reality_pairs = [
            ("sponge", "rock", "looks like a rock but is actually a sponge"),
            ("candle", "apple", "looks like an apple but is actually a candle"),
            ("soap", "chocolate", "looks like chocolate but is actually soap"),
            ("eraser", "candy", "looks like candy but is actually an eraser")
        ]

        # Expected cognitive actions for ToM
        self.tom_cognitive_actions = [
            "perspective_taking",
            "hypothesis_generation",
            "metacognitive_monitoring",
            "distinguishing",
            "updating_beliefs",
            "counterfactual_reasoning",
            "suspending_judgment"
        ]

    def generate_task_id(self, task_type: ToMTaskType) -> str:
        """Generate unique task ID"""
        self.task_count += 1
        return f"{task_type.value}_{self.task_count:04d}"

    def generate_false_belief_task(self, difficulty: TaskDifficulty = TaskDifficulty.MEDIUM) -> ToMTask:
        """
        Generate a Sally-Anne style false belief task

        Classic structure:
        1. Character A places object in location X
        2. Character A leaves
        3. Character B moves object to location Y
        4. Character A returns
        5. Question: Where will Character A look for the object?

        Correct answer: Location X (where they believe it is)
        Incorrect answer: Location Y (where it actually is - reality bias)
        """
        char_a = random.choice(self.characters)
        char_b = random.choice([c for c in self.characters if c != char_a])
        item = random.choice(self.items)
        loc_x = random.choice(self.containers)
        loc_y = random.choice([c for c in self.containers if c != loc_x])
        setting = random.choice(self.locations)

        # Main scenario requiring ToM
        scenario = (
            f"{char_a} and {char_b} are in the {setting}. "
            f"{char_a} puts a {item} in the {loc_x} and then leaves to go outside. "
            f"While {char_a} is gone, {char_b} takes the {item} out of the {loc_x} "
            f"and puts it in the {loc_y}. "
            f"{char_a} comes back inside."
        )

        # Control scenario (no mental states, just physical facts)
        control_scenario = (
            f"{char_a} and {char_b} are in the {setting}. "
            f"A {item} is first in the {loc_x}. "
            f"Then the {item} is moved to the {loc_y}. "
            f"{char_a} was not present when the {item} was moved."
        )

        question = f"Where will {char_a} look for the {item}?"
        correct_answer = f"in the {loc_x}"

        # Reality bias alternative (common error)
        alternatives = [
            f"in the {loc_y}",
            f"in both the {loc_x} and {loc_y}",
            f"{char_a} won't look for it"
        ]

        if difficulty == TaskDifficulty.HARD:
            # Add more complex language and distractors
            scenario += f" {char_b} tells {char_a} that the {setting} has been reorganized."
            alternatives.append(f"anywhere in the {setting}")

        return ToMTask(
            task_id=self.generate_task_id(ToMTaskType.FALSE_BELIEF),
            task_type=ToMTaskType.FALSE_BELIEF,
            difficulty=difficulty,
            scenario=scenario,
            control_scenario=control_scenario,
            question=question,
            correct_answer=correct_answer,
            alternatives=alternatives,
            characters=[char_a, char_b],
            objects=[item],
            locations=[loc_x, loc_y],
            expected_cognitive_actions=self.tom_cognitive_actions[:5],
            tom_explanation=(
                f"Requires tracking that {char_a} has a false belief about the {item}'s location. "
                f"The reasoner must distinguish between reality (item in {loc_y}) and {char_a}'s "
                f"belief (thinks item is still in {loc_x})."
            )
        )

    def generate_unexpected_contents_task(self, difficulty: TaskDifficulty = TaskDifficulty.MEDIUM) -> ToMTask:
        """
        Generate Smarties-style unexpected contents task

        Structure:
        1. Show container with typical label (e.g., candy box)
        2. Reveal unexpected contents (e.g., pencils instead)
        3. Ask what a naive person would think is inside before seeing

        Tests understanding that others don't know what you know
        """
        container, (expected, actual, *alternatives_pool) = random.choice(list(self.container_items.items()))
        char = random.choice(self.characters)

        scenario = (
            f"You see a {container} on the table. Before opening it, it looks like a normal {container}. "
            f"You open it and find {actual} inside, not {expected}. "
            f"Your friend {char} has never seen inside this {container} before. "
            f"{char} walks into the room and sees the closed {container}."
        )

        control_scenario = (
            f"A {container} is on the table. The {container} contains {actual}. "
            f"The {container} is labeled as containing {expected}. "
            f"{char} sees the closed {container}."
        )

        question = f"What does {char} think is inside the {container}?"
        correct_answer = expected
        alternatives = [actual] + random.sample(alternatives_pool, 2)

        if difficulty == TaskDifficulty.HARD:
            # Add complexity: multiple people with different knowledge states
            char2 = random.choice([c for c in self.characters if c != char])
            scenario += f" Earlier, {char2} looked inside the {container}."
            question = f"What does {char} think is inside? What does {char2} think is inside?"

        return ToMTask(
            task_id=self.generate_task_id(ToMTaskType.UNEXPECTED_CONTENTS),
            task_type=ToMTaskType.UNEXPECTED_CONTENTS,
            difficulty=difficulty,
            scenario=scenario,
            control_scenario=control_scenario,
            question=question,
            correct_answer=correct_answer,
            alternatives=alternatives,
            characters=[char],
            objects=[container, expected, actual],
            locations=["table"],
            expected_cognitive_actions=["perspective_taking", "distinguishing", "metacognitive_monitoring"],
            tom_explanation=(
                f"Requires understanding that {char} doesn't know what you know. "
                f"{char} will have the default belief ({expected}) based on the container's appearance, "
                f"not your privileged knowledge ({actual})."
            )
        )

    def generate_appearance_reality_task(self, difficulty: TaskDifficulty = TaskDifficulty.MEDIUM) -> ToMTask:
        """
        Generate appearance-reality task

        Tests ability to distinguish between how things appear vs. what they are
        Classic ToM test - requires representing two different perspectives simultaneously
        """
        actual, appears_as, description = random.choice(self.appearance_reality_pairs)
        char = random.choice(self.characters)

        scenario = (
            f"There is an object on the table that {description}. "
            f"When you look at it closely, you realize it's actually a {actual}, "
            f"even though it looks exactly like a {appears_as}. "
            f"{char} has not touched or examined the object closely. "
            f"{char} is just looking at it from across the room."
        )

        control_scenario = (
            f"An object on the table is a {actual}. "
            f"The object resembles a {appears_as}. "
            f"{char} is looking at the object from across the room."
        )

        question = f"What does {char} think the object is?"
        correct_answer = appears_as
        alternatives = [actual, f"both a {actual} and a {appears_as}", "neither"]

        if difficulty == TaskDifficulty.HARD:
            scenario += f" However, {char} is known for being very observant and skeptical."
            alternatives.append(f"{char} would be suspicious and uncertain")

        return ToMTask(
            task_id=self.generate_task_id(ToMTaskType.APPEARANCE_REALITY),
            task_type=ToMTaskType.APPEARANCE_REALITY,
            difficulty=difficulty,
            scenario=scenario,
            control_scenario=control_scenario,
            question=question,
            correct_answer=correct_answer,
            alternatives=alternatives,
            characters=[char],
            objects=[actual, appears_as],
            locations=["table"],
            expected_cognitive_actions=[
                "perspective_taking", "distinguishing", "suspending_judgment", "metacognitive_monitoring"
            ],
            tom_explanation=(
                f"Requires representing {char}'s perspective (sees {appears_as}) "
                f"while maintaining your own knowledge (knows it's {actual}). "
                f"Tests ability to hold dual representations simultaneously."
            )
        )

    def generate_second_order_belief_task(self, difficulty: TaskDifficulty = TaskDifficulty.HARD) -> ToMTask:
        """
        Generate second-order false belief task

        Structure: A thinks that B thinks X
        Most complex ToM task - requires recursive mental state attribution
        """
        char_a = random.choice(self.characters)
        char_b = random.choice([c for c in self.characters if c != char_a])
        char_c = random.choice([c for c in self.characters if c not in [char_a, char_b]])
        item = random.choice(self.items)
        loc_x = random.choice(self.locations)
        loc_y = random.choice([l for l in self.locations if l != loc_x])

        scenario = (
            f"{char_a} and {char_b} are together in the {loc_x}. "
            f"{char_a} tells {char_b} that the {item} is in the {loc_x}. "
            f"{char_b} leaves. While {char_b} is gone, {char_c} moves the {item} to the {loc_y} "
            f"and tells {char_a} about it. {char_a} sees the {item} in the new location. "
            f"{char_b} doesn't know the {item} was moved. "
            f"{char_a} doesn't know that {char_b} doesn't know about the move."
        )

        control_scenario = (
            f"Initially, {char_a} and {char_b} both knew the {item} was in the {loc_x}. "
            f"The {item} was moved to the {loc_y}. "
            f"{char_a} knows about the move. {char_b} does not know about the move."
        )

        question = f"Where does {char_a} think that {char_b} will look for the {item}?"
        correct_answer = f"in the {loc_x}"
        alternatives = [
            f"in the {loc_y}",
            f"{char_a} doesn't know where {char_b} will look",
            f"in both locations"
        ]

        return ToMTask(
            task_id=self.generate_task_id(ToMTaskType.SECOND_ORDER_BELIEF),
            task_type=ToMTaskType.SECOND_ORDER_BELIEF,
            difficulty=difficulty,
            scenario=scenario,
            control_scenario=control_scenario,
            question=question,
            correct_answer=correct_answer,
            alternatives=alternatives,
            characters=[char_a, char_b, char_c],
            objects=[item],
            locations=[loc_x, loc_y],
            expected_cognitive_actions=[
                "perspective_taking", "hypothesis_generation", "metacognitive_monitoring",
                "distinguishing", "counterfactual_reasoning", "updating_beliefs"
            ],
            tom_explanation=(
                f"Requires recursive reasoning: {char_a} thinks that {char_b} thinks the {item} is in {loc_x}. "
                f"Must track multiple nested belief states and distinguish who knows what."
            )
        )

    def generate_affective_tom_task(self, difficulty: TaskDifficulty = TaskDifficulty.MEDIUM) -> ToMTask:
        """
        Generate affective ToM task (emotion/motivation inference)

        Tests ability to infer others' emotions and motivations from situations
        """
        char = random.choice(self.characters)

        scenarios_pool = [
            {
                "scenario": f"{char} studied hard for weeks for an important test. On test day, {char} arrived late and missed the test entirely.",
                "emotion": "disappointed and upset",
                "alternatives": ["happy and relieved", "angry at the teacher", "indifferent"],
                "objects": ["test"],
                "locations": ["school"]
            },
            {
                "scenario": f"{char} spent months planning a surprise birthday party for their best friend. When the friend arrived, they smiled widely and hugged everyone.",
                "emotion": "happy and grateful",
                "alternatives": ["confused and suspicious", "sad and lonely", "angry and betrayed"],
                "objects": ["birthday party"],
                "locations": ["party venue"]
            },
            {
                "scenario": f"{char} was promised they could go to the amusement park this weekend. At the last minute, their parents said they had to cancel the trip.",
                "emotion": "disappointed and frustrated",
                "alternatives": ["excited and eager", "grateful and relieved", "proud and accomplished"],
                "objects": ["amusement park trip"],
                "locations": ["home"]
            },
            {
                "scenario": f"{char} found their lost pet after searching for three days. The pet ran up to {char} wagging its tail.",
                "emotion": "relieved and joyful",
                "alternatives": ["angry and resentful", "scared and nervous", "indifferent"],
                "objects": ["pet"],
                "locations": ["neighborhood"]
            }
        ]

        scenario_data = random.choice(scenarios_pool)
        scenario_text = scenario_data["scenario"]
        correct_emotion = scenario_data["emotion"]
        emotion_alternatives = scenario_data["alternatives"]

        control_scenario = scenario_text  # Same scenario, but control asks factual question

        question = f"How does {char} likely feel in this situation?"
        correct_answer = correct_emotion
        alternatives = emotion_alternatives

        if difficulty == TaskDifficulty.HARD:
            # Add conflicting cues
            scenario_text += f" However, {char} tried to hide their true feelings and smiled."
            alternatives.append("pretending to feel differently than they actually do")

        return ToMTask(
            task_id=self.generate_task_id(ToMTaskType.AFFECTIVE_TOM),
            task_type=ToMTaskType.AFFECTIVE_TOM,
            difficulty=difficulty,
            scenario=scenario_text,
            control_scenario=control_scenario,
            question=question,
            correct_answer=correct_answer,
            alternatives=alternatives,
            characters=[char],
            objects=scenario_data["objects"],
            locations=scenario_data["locations"],
            expected_cognitive_actions=[
                "emotion_perception", "perspective_taking", "hypothesis_generation",
                "emotion_understanding", "connecting"
            ],
            tom_explanation=(
                f"Requires inferring {char}'s emotional state from the situation. "
                f"Must understand how events affect mental/emotional states of others."
            )
        )

    def generate_task_suite(
        self,
        n_false_belief: int = 30,
        n_unexpected: int = 20,
        n_appearance: int = 20,
        n_second_order: int = 15,
        n_affective: int = 20
    ) -> List[ToMTask]:
        """Generate full suite of ToM tasks with difficulty distribution"""
        tasks = []

        # False Belief tasks
        for i in range(n_false_belief):
            difficulty = self._get_difficulty_for_index(i, n_false_belief)
            tasks.append(self.generate_false_belief_task(difficulty))

        # Unexpected Contents tasks
        for i in range(n_unexpected):
            difficulty = self._get_difficulty_for_index(i, n_unexpected)
            tasks.append(self.generate_unexpected_contents_task(difficulty))

        # Appearance-Reality tasks
        for i in range(n_appearance):
            difficulty = self._get_difficulty_for_index(i, n_appearance)
            tasks.append(self.generate_appearance_reality_task(difficulty))

        # Second-Order Belief tasks (all hard)
        for i in range(n_second_order):
            tasks.append(self.generate_second_order_belief_task(TaskDifficulty.HARD))

        # Affective ToM tasks
        for i in range(n_affective):
            difficulty = self._get_difficulty_for_index(i, n_affective)
            tasks.append(self.generate_affective_tom_task(difficulty))

        return tasks

    def _get_difficulty_for_index(self, index: int, total: int) -> TaskDifficulty:
        """Distribute difficulty levels across tasks"""
        ratio = index / total
        if ratio < 0.5:
            return TaskDifficulty.EASY
        elif ratio < 0.8:
            return TaskDifficulty.MEDIUM
        else:
            return TaskDifficulty.HARD

    def save_tasks(self, tasks: List[ToMTask], output_path: Path):
        """Save tasks to JSON file"""
        tasks_dict = [asdict(task) for task in tasks]
        # Convert enums to strings
        for task_dict in tasks_dict:
            task_dict['task_type'] = task_dict['task_type'].value
            task_dict['difficulty'] = task_dict['difficulty'].value

        with open(output_path, 'w') as f:
            json.dump(tasks_dict, f, indent=2)

        print(f"Saved {len(tasks)} ToM tasks to {output_path}")

    def load_tasks(self, input_path: Path) -> List[ToMTask]:
        """Load tasks from JSON file"""
        with open(input_path, 'r') as f:
            tasks_dict = json.load(f)

        tasks = []
        for task_dict in tasks_dict:
            # Convert strings back to enums
            task_dict['task_type'] = ToMTaskType(task_dict['task_type'])
            task_dict['difficulty'] = TaskDifficulty(task_dict['difficulty'])
            tasks.append(ToMTask(**task_dict))

        return tasks


def main():
    """Generate and save default ToM task suite"""
    generator = ToMTaskGenerator(seed=42)

    # Generate full suite
    print("Generating Theory of Mind task suite...")
    tasks = generator.generate_task_suite(
        n_false_belief=30,
        n_unexpected=20,
        n_appearance=20,
        n_second_order=15,
        n_affective=20
    )

    print(f"\nGenerated {len(tasks)} total tasks:")
    task_type_counts = {}
    for task in tasks:
        task_type = task.task_type.value
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    for task_type, count in sorted(task_type_counts.items()):
        print(f"  {task_type}: {count}")

    # Save to file
    output_path = Path(__file__).parent.parent.parent / "data" / "tom_tasks" / "tom_task_suite.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_tasks(tasks, output_path)

    # Print example tasks
    print("\n" + "="*80)
    print("EXAMPLE TASKS")
    print("="*80)

    for task_type in ToMTaskType:
        example_task = next((t for t in tasks if t.task_type == task_type), None)
        if example_task:
            print(f"\n{task_type.value.upper().replace('_', ' ')}:")
            print(f"Task ID: {example_task.task_id}")
            print(f"Difficulty: {example_task.difficulty.value}")
            print(f"\nScenario:\n{example_task.scenario}")
            print(f"\nQuestion: {example_task.question}")
            print(f"Correct Answer: {example_task.correct_answer}")
            print(f"Alternatives: {', '.join(example_task.alternatives)}")
            print(f"\nToM Explanation: {example_task.tom_explanation}")
            print(f"Expected Cognitive Actions: {', '.join(example_task.expected_cognitive_actions)}")
            print("-" * 80)


if __name__ == "__main__":
    main()
