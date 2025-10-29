I would need to steer the model towards ToM using repeng library that is available in the repo ToM folder.
So lets analyse how it was implemented there and lets steer gemma 3 4b, I would need to steer it towards being better at theory of mind, so the idea is to train that vector, vector that steers the model towards being better at ToM. 



## Simple ToM Steering Approach

### Core Pattern (from repeng examples)

Use the established repeng pattern:

```python
# Simple persona-based steering for ToM
tom_dataset = make_dataset(
    "Act as if you're extremely good at {persona}.",
    [
        "understanding what others are thinking and feeling",
        "inferring intentions and beliefs of other people", 
        "predicting what people will do based on their mental states",
        "perspective-taking and empathy",
        "tracking what others know and don't know"
    ],
    [
        "not understanding what others are thinking and feeling",
        "struggling to infer intentions and beliefs",
        "having trouble predicting actions from mental states",
        "lacking perspective-taking abilities",
        "confusing what others know vs what they don't know"
    ],
    truncated_output_suffixes,  # or use any conversation snippets
)
```

### Alternative: Single Template Versions

```python
# Option 1: Inferring intentions
intention_dataset = make_dataset(
    "Pretend you're someone who {persona}.",
    ["excels at inferring intentions and understanding what others are thinking"],
    ["struggles to understand what others are thinking or feeling"],
    suffixes
)

# Option 2: Social reasoning
social_dataset = make_dataset(
    "You are {persona} at understanding and predicting human behavior.",
    ["exceptionally skilled"],
    ["not skilled"],
    suffixes
)

# Option 3: Perspective taking
perspective_dataset = make_dataset(
    "Act as if you're {persona} at perspective-taking and empathy.",
    ["extremely good"],
    ["very poor"],
    suffixes
)
```

### Implementation Code

```python
import json
from repeng import ControlVector, ControlModel, DatasetEntry

def make_dataset(template: str, pos_personas: list[str], neg_personas: list[str], suffixes: list[str]):
    dataset = []
    for suffix in suffixes:
        for positive_persona, negative_persona in zip(pos_personas, neg_personas):
            dataset.append(
                DatasetEntry(
                    positive=template.format(persona=positive_persona) + suffix,
                    negative=template.format(persona=negative_persona) + suffix,
                )
            )
    return dataset

# Load model
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
model = ControlModel(model, list(range(-5, -18, -1)))

# Load suffixes (conversation continuations from your data)
with open("data/all_truncated_outputs.json") as f:
    output_suffixes = json.load(f)

## Use this one -- BEST ONE
# Create ToM dataset with simple persona contrast
tom_dataset = make_dataset(
    "You're {persona}.",
    ["excellent at understanding minds, predicting behavior, and inferring what others think and feel"],
    ["terrible at understanding minds, predicting behavior, and inferring what others think and feel"],
    output_suffixes
)

# Train the vector
model.reset()
tom_vector = ControlVector.train(model, tokenizer, tom_dataset)

# Use it
model.set_control(tom_vector, strength=1.5)
```

Approach: steer toward "social reasoning" or "mind reading" personas using the same pattern as repeng happiness/honesty/self-awareness.

or something like this

```python
# More aligned with BigToM evaluation
tom_dataset = make_dataset(
    "You're {persona} at tracking what others know and believe.",
    [
        "excellent",  # good at ToM
    ],
    [
        "poor",  # bad at ToM  
    ],
    suffixes
)

# Or multiple specific skills:
belief_tracking_dataset = make_dataset(
    "You {persona} track what others believe vs what's actually true.",
    ["accurately"],  # maintains false beliefs when appropriate
    ["confuse"]     # mixes up what's true vs what character believes
)
```

```python
import json
from repeng import ControlVector, ControlModel, DatasetEntry

def make_dataset(template: str, pos_personas: list[str], neg_personas: list[str], suffixes: list[str]):
    """Helper function to create dataset entries"""
    dataset = []
    for suffix in suffixes:
        for positive_persona, negative_persona in zip(pos_personas, neg_personas):
            dataset.append(
                DatasetEntry(
                    positive=template.format(persona=positive_persona) + suffix,
                    negative=template.format(persona=negative_persona) + suffix,
                )
            )
    return dataset


# Load suffixes (use your existing data)
with open("data/all_truncated_outputs.json") as f:
    output_suffixes = json.load(f)


# ===== PROMPT 1: forward_belief =====
# Skill: Predicting what someone believes based on what they know/don't know
forward_belief_dataset = make_dataset(
    "You're {persona} at figuring out what people believe given what they've seen or haven't seen.",
    [
        "accurate at maintaining false beliefs when someone didn't see a change",
        "precise at understanding people maintain outdated beliefs when unaware",
        "skilled at tracking what someone believes vs what's actually true"
    ],
    [
        "always confused between what people believe and what's actually true",
        "incorrectly updating beliefs even when someone didn't witness changes",
        "unable to distinguish between what someone knows vs doesn't know"
    ],
    output_suffixes
)


# ===== PROMPT 2: forward_action =====
# Skill: Predicting what someone will do based on their beliefs and desires
forward_action_dataset = make_dataset(
    "You {persona} predict what people will do based on their beliefs and what they want.",
    [
        "correctly predict actions that follow from desires and beliefs",
        "understand that people act on beliefs, not necessarily reality",
        "accurately infer actions by connecting desires → beliefs → actions"
    ],
    [
        "predict actions based only on reality, ignoring beliefs",
        "can't connect desires and beliefs to predict behaviors",
        "think people always act on complete accurate information"
    ],
    output_suffixes
)


# ===== PROMPT 3: backward_belief =====
# Skill: Inferring what someone believes from their actions
backward_belief_dataset = make_dataset(
    "When you see what someone does, you {persona} infer what they believe.",
    [
        "correctly infer beliefs from observed actions",
        "accurately reason backward from behaviors to mental states",
        "understand actions reveal beliefs"
    ],
    [
        "can't reason backward from actions to beliefs",
        "ignore actions when inferring what people think",
        "don't realize actions provide clues about beliefs"
    ],
    output_suffixes
)


# ===== PROMPT 4: percept_to_belief =====
# Skill: Mapping what someone sees/perceives to what they believe
percept_to_belief_dataset = make_dataset(
    "You {persona} connect what people perceive to what they come to believe.",
    [
        "accurately map perceptions to beliefs",
        "understand that perception leads to belief formation",
        "correctly reason: perceive → believe"
    ],
    [
        "fail to connect perception to belief",
        "can't map what people see to what they think",
        "don't understand perception cues indicate beliefs"
    ],
    output_suffixes
)


# ===== TRAIN INDIVIDUAL VECTORS =====
# You can train each separately or combine them

# Train forward_belief vector
model.reset()
forward_belief_vector = ControlVector.train(
    model, tokenizer, forward_belief_dataset, method="pca_diff"
)

# Train forward_action vector  
model.reset()
forward_action_vector = ControlVector.train(
    model, tokenizer, forward_action_dataset, method="pca_diff"
)

# Train backward_belief vector
model.reset()
backward_belief_vector = ControlVector.train(
    model, tokenizer, backward_belief_dataset, method="pca_diff"
)

# Train percept_to_belief vector
model.reset()
percept_to_belief_vector = ControlVector.train(
    model, tokenizer, percept_to_belief_dataset, method="pca_diff"
)


# ===== COMBINE INTO ONE VECTOR =====
# You can combine all vectors together
combined_tom_vector = (
    forward_belief_vector +
    forward_action_vector + 
    backward_belief_vector +
    percept_to_belief_vector
) / 4  # Average or weight differently

# Or apply separately with different strengths
# model.set_control(forward_belief_vector, 1.5)
# model.set_control(forward_action_vector, 1.5) 
# etc.
```


### Skills by axis

- **Order init (0 vs 1)**
  - 0 (implicit belief): infer the agent’s belief from context; stronger perspective-taking, discourse/model-building, and inhibition of your own knowledge.
  - 1 (explicit belief stated): bind and track a stated belief over subsequent events; working memory and consistency checking.

- **Direction (forward vs backward)**
  - Forward: maintain a situation model as events unfold; temporal tracking; predict beliefs/actions from current context.
  - Backward: abductive reasoning from later observations to earlier mental states; causal-temporal reconstruction; plan/outcome interpretation.

- **Variable (belief vs action)**
  - Belief: first-order Theory of Mind—represent what the agent believes vs what is true; perspective decoupling.
  - Action: plan/action prediction conditioned on the agent’s belief and goal; goal-conditioned policy simulation.

- **Belief type (true_belief vs false_belief)**
  - True: align agent belief with world state; consistency and factual tracking.
  - False: decouple agent belief from reality; counterfactual simulation (“what they think” vs “what is”); stronger inhibition of evaluator knowledge.

### Skills per condition family

- **Forward belief — true_belief**
  - Track unfolding facts; attribute correct belief; maintain consistency.
- **Forward belief — false_belief**
  - Infer belief that conflicts with reality; suppress your own knowledge; classic first-order ToM with misbelief.
- **Backward belief — true_belief**
  - Reason backward from later cues/actions to reconstruct a correct prior belief; causal/abductive inference.
- **Backward belief — false_belief**
  - Abduct prior misbelief from later evidence; distinguish outcomes driven by ignorance vs world changes; robust perspective-taking.
- **Forward action — true_belief**
  - Predict goal-directed action from accurate beliefs; commonsense planning.
- **Forward action — false_belief**
  - Predict action from the agent’s mistaken belief; simulate plans under incorrect world models; strong belief–action decoupling.

### Core capabilities exercised across tests
- Reading comprehension and event sequencing
- Working memory for entities, states, and updates
- Causal-temporal reasoning and abduction
- First-order Theory of Mind and perspective decoupling
- Counterfactual simulation and knowledge inhibition
- Goal recognition and commonsense planning