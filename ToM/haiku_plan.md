# Comprehensive Plan: Training Multiple Theory of Mind Steering Vectors with CAA
```python
PERSONA_TEMPLATES = {
    "core_tom": {
        "name": "Core Theory of Mind",
        "system_instruction": "Act as if you have {persona}.",
        "positive": [
            "an exceptional ability to take others' perspectives and simulate their mental states",
            "a strong capacity for counterfactual reasoning about what others believe to be true",
            "a remarkable talent for understanding how multiple agents' beliefs interact and diverge"
        ],
        "negative": [
            "a very limited ability to understand how others think and feel",
            "a poor capacity for inhibiting your own knowledge to understand another's perspective",
            "a fundamental confusion about the difference between your own knowledge and what others know"
        ]
    },
    
    "forward_belief_true": {
        "name": "Forward Belief - True Belief",
        "system_instruction": "You track what people believe as events happen {persona}.",
        "positive": [
            "with high accuracy, updating their beliefs based on new information they receive",
            "exceptionally well, recognizing when an agent has observed information and therefore holds an accurate belief",
            "precisely, understanding that when people see events, their beliefs align with reality"
        ],
        "negative": [
            "poorly, struggling to understand that a person's knowledge is updated by what they observe",
            "incorrectly, failing to recognize that seeing something guarantees knowledge of it",
            "inaccurately, often doubting beliefs that are actually grounded in direct observation"
        ]
    },
    
    "forward_belief_false": {
        "name": "Forward Belief - False Belief",
        "system_instruction": "You track what people believe as events happen {persona}.",
        "positive": [
            "exceptionally well, recognizing when an agent has missed information and therefore holds a false belief",
            "with excellence, understanding that people can believe things that contradict reality if they lack key information",
            "skillfully, recognizing the difference between what happened and what the agent thinks happened"
        ],
        "negative": [
            "poorly, struggling to understand that a person's knowledge is limited to what they have observed",
            "inaccurately, always assuming agents know the current truth regardless of what they've seen",
            "incorrectly, failing to grasp that unawareness creates false beliefs"
        ]
    },
    
    "backward_belief": {
        "name": "Backward Belief (Abductive Reasoning)",
        "system_instruction": "You infer past beliefs from current evidence {persona}.",
        "positive": [
            "skillfully, reconstructing what someone must have believed by looking at their later actions",
            "with excellence, determining if an outcome was due to a prior false belief or a change in the world",
            "expertly, working backwards from behavior to identify the underlying mental state that caused it"
        ],
        "negative": [
            "poorly, unable to work backward from an action to understand the belief that caused it",
            "inaccurately, failing to distinguish between ignorance and a changed reality when explaining events",
            "confusingly, often confusing what the person knew with what they did"
        ]
    },
    
    "forward_action_true": {
        "name": "Forward Action - True Belief",
        "system_instruction": "At predicting what people will do, you are {persona}.",
        "positive": [
            "able to simulate the plan an agent will follow based on their true beliefs about the world, with excellent accuracy",
            "especially adept at predicting actions when an agent's beliefs accurately reflect reality",
            "skilled at understanding that accurate beliefs lead to rational, goal-aligned actions"
        ],
        "negative": [
            "unable to predict actions based on how agents with accurate beliefs will behave, performing poorly",
            "struggling to connect true beliefs with the logical actions they produce, showing a lack of skill",
            "often confused, predicting actions that contradict what someone with accurate beliefs would do"
        ]
    },
    
    "forward_action_false": {
        "name": "Forward Action - False Belief",
        "system_instruction": "At predicting what people will do, you are {persona}.",
        "positive": [
            "able to simulate the plan an agent will follow based on their potentially false beliefs about the world, with excellent insight",
            "especially adept at predicting actions that logically follow from an agent's mistaken beliefs",
            "skilled at understanding that even false beliefs drive coherent, goal-directed behavior"
        ],
        "negative": [
            "able to predict actions only based on the actual state of the world while ignoring individual beliefs, performing poorly",
            "assuming that people always act with perfect and complete information, showing a lack of skill",
            "confused, predicting what someone should do instead of what their false beliefs would lead them to do"
        ]
    }
}
```



## Recommended Workflow

Here's how I'd restructure your training data pipeline:

### 1. Extract Scenario-Answer Pairs from BigToM

From your CSV, isolate:
- Scenarios (the narrative)
- Correct answers
- Incorrect answers (which you already have!)

### 2. Create CAA Training Dataset

```python
def create_tom_caa_dataset(scenarios_with_answers):
    """
    Transform evaluation data into CAA training data
    """
    caa_pairs = []
    
    for scenario_row in scenarios_with_answers:
        scenario_text = scenario_row['scenario']
        question = scenario_row['question']
        correct_answer = scenario_row['correct_answer']
        incorrect_answer = scenario_row['incorrect_answer']
        
        # Map scenario to its ToM skill type
        skill_type = classify_skill(scenario_row)  # "forward_belief", "backward_belief", etc.
        
        # Get the appropriate persona template and prompts
        personas = get_personas(skill_type)
        
        # Create positive example (excellent performance)
        positive_prompt = f"""You are {personas['positive_system']}.

{scenario_text}

{question}"""
        
        positive_completion = correct_answer
        
        # Create negative example (poor performance)  
        negative_prompt = f"""You are {personas['negative_system']}.

{scenario_text}

{question}"""
        
        negative_completion = incorrect_answer
        
        caa_pairs.append({
            'skill': skill_type,
            'positive_prompt': positive_prompt,
            'positive_completion': positive_completion,
            'negative_prompt': negative_prompt,
            'negative_completion': negative_completion,
            'scenario': scenario_text
        })
    
    return caa_pairs
```

### 3. Map Scenarios to Skill Types

Your BigToM data has conditions. Map them:

```python
def classify_skill(scenario_row):
    """
    The repo generates scenarios with labels like:
    - forward_belief_true_belief
    - forward_belief_false_belief
    - backward_belief_true_belief
    etc.
    """
    scenario_text = scenario_row['scenario']
    
    # Look for signals in the scenario structure
    if "notices" in scenario_text or "sees" in scenario_text:
        if "mistakenly" in scenario_text or "unknowingly" in scenario_text:
            return "forward_belief_false_belief"
        else:
            return "forward_belief_true_belief"
    
    # This would come from BigToM's generate_conditions.py output
    return scenario_row.get('condition_type', 'core_tom')
```

### 4. Use Skill-Specific Personas

Here's how to map your document's personas to the actual data:


```


## Key Insights from Your Document

The persona approach in your document is **cleverly designed**:

1. **Variable (Belief vs. Action)**: Uses adjectives + clarifying clauses → good for distinguishing distinct concepts
2. **Belief Type (True vs. False)**: Specifically targets false belief reasoning
3. **Forward Belief**: Adverbial phrases describing **how** tracking happens → focuses on process
4. **Backward Belief**: Abductive reasoning about **past** mental states → different skill than forward
5. **Forward Action**: Prediction based on beliefs
6. **Core Capabilities**: General catch-all for broad ToM

## Recommended Data Split

- **70%**: Forward Belief (True + False) - this is where models struggle most
- **15%**: Forward Action - critical for planning tasks
- **10%**: Backward Belief - useful but less common
- **5%**: Core ToM - general capability boost

This gives you ~several thousand contrastive pairs if BigToM has ~1000-2000 scenarios, which is solid for CAA training. Would you like me to help you write the actual data pipeline script?