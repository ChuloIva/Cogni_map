Ah, you're absolutely right! You need a **prompt template system** that programmatically generates unique prompts for each call. Here's how to build that:

## PROMPT GENERATION SYSTEM

### Core Architecture

```python
# Pseudocode structure
templates = load_templates()
variables = load_variable_pools()

for i in range(100000):
    # Randomly select components
    cognitive_action = random.choice(cognitive_actions)
    domain = random.choice(domains)
    context_detail = random.choice(context_details[domain])
    prompt_template = random.choice(templates[cognitive_action])
    
    # Inject variables into template
    prompt = prompt_template.format(
        domain=domain,
        context=context_detail,
        subject=random.choice(subjects),
        emotional_state=random.choice(emotional_states),
        trigger=random.choice(triggers),
        etc...
    )
    
    # Add uniqueness constraints
    prompt += f"\n\nMake this example unique by focusing on: {random.choice(unique_angles)}"
    prompt += f"\nUse language style: {random.choice(language_styles)}"
    prompt += f"\nExample #{i+1} of this batch."
    
    # Send to LLM
    response = llm.generate(prompt)
```

## COMPLETE PROMPT TEMPLATE LIBRARY

### Template Structure

Each prompt should combine:
1. **Cognitive action** (what to demonstrate)
2. **Specific scenario** (concrete situation)
3. **Constraint/variation** (what makes THIS example unique)
4. **Format instruction** (how to structure it)

---

## MASTER PROMPT TEMPLATES

### For Single Cognitive Actions

```
TEMPLATE 1: Scenario-Based
Generate 1 example (2-4 sentences) showing someone {COGNITIVE_ACTION} in this specific scenario:

Scenario: {SUBJECT} is {SITUATION} and experiences {TRIGGER_EVENT}. They engage in {COGNITIVE_ACTION}.

Requirements:
- Domain: {DOMAIN}
- Emotional context: {EMOTIONAL_STATE}
- Show the cognitive process explicitly
- Use {LANGUAGE_STYLE} language
- Focus angle: {UNIQUE_ANGLE}
- Do NOT use these words: {FORBIDDEN_WORDS}

Output only the example text, no preamble.
```

```
TEMPLATE 2: Before/After
Generate 1 example (2-4 sentences) showing {COGNITIVE_ACTION} by contrasting before and after:

Context: {DOMAIN} situation involving {CONTEXT_DETAIL}
Before state: {INITIAL_STATE}
Cognitive action: {COGNITIVE_ACTION}
After state: {RESULT_STATE}

Requirements:
- Subject: {SUBJECT}
- Complexity: {COMPLEXITY_LEVEL}
- Include specific details about {FOCUS_ELEMENT}
- Avoid cliché phrasings
- Language register: {REGISTER}

Output only the example text, no preamble.
```

```
TEMPLATE 3: Process-Focused
Generate 1 example (2-4 sentences) that shows the PROCESS of {COGNITIVE_ACTION}:

Setup: {SUBJECT} faces {PROBLEM_TYPE} related to {DOMAIN}
The cognitive action unfolds through: {PROCESS_STAGES}

Requirements:
- Make the internal process visible
- Include: {REQUIRED_ELEMENT}
- Perspective: {PERSPECTIVE_TYPE}
- Incorporate this complication: {COMPLICATION}
- Avoid these phrases: {AVOID_PHRASES}

Output only the example text, no preamble.
```

```
TEMPLATE 4: Trigger-Response
Generate 1 example (2-4 sentences) where {TRIGGER} leads to {COGNITIVE_ACTION}:

Context: {CONTEXT_DETAIL} in {DOMAIN}
Trigger: {SPECIFIC_TRIGGER}
Cognitive response: {COGNITIVE_ACTION}
Outcome direction: {OUTCOME_DIRECTION}

Requirements:
- Subject age/type: {SUBJECT_TYPE}
- Time frame: {TIME_FRAME}
- Include sensory or emotional detail
- Unique angle: {UNIQUE_CONSTRAINT}

Output only the example text, no preamble.
```

---

## VARIABLE POOLS (What fills the {PLACEHOLDERS})

### Cognitive Actions (20 core)
```python
cognitive_actions = [
    "reconsidering a belief",
    "reframing a situation", 
    "noticing a pattern",
    "taking another's perspective",
    "questioning an assumption",
    "abstracting from specifics",
    "making something concrete",
    "connecting disparate ideas",
    "distinguishing between concepts",
    "updating a mental model",
    "suspending judgment",
    "recognizing a recurring pattern",
    "zooming out for context",
    "focusing on details",
    "reappraising emotions",
    "drawing an analogy",
    "counterfactual reasoning",
    "generating hypotheses",
    "meta-cognitive reflection",
    "accepting and letting go"
]
```

### Subjects (vary who's doing the thinking)
```python
subjects = [
    "a software developer",
    "a parent",
    "a student",
    "a manager",
    "a therapist",
    "an artist",
    "a researcher",
    "someone in their 20s",
    "someone in their 50s",
    "a team leader",
    "an entrepreneur",
    "a teacher",
    "a friend",
    "a partner in a relationship",
    "someone reflecting alone",
    "a person in therapy",
    "a writer",
    "a scientist",
    "a career changer",
    "someone grieving"
    # ... add 50+ more
]
```

### Domains (20+)
```python
domains = [
    "personal relationships",
    "career decisions",
    "creative work",
    "scientific research",
    "moral dilemmas",
    "health choices",
    "financial planning",
    "learning and education",
    "conflict resolution",
    "identity and self-concept",
    "parenting",
    "communication challenges",
    "goal achievement",
    "dealing with failure",
    "processing success",
    "daily decisions",
    "philosophical questions",
    "team dynamics",
    "time management",
    "personal growth"
    # ... add more
]
```

### Context Details (specific scenarios for each domain)
```python
context_details = {
    "personal relationships": [
        "after a difficult conversation with a partner",
        "noticing a friendship pattern",
        "considering whether to reconnect with someone",
        "processing feedback from a family member",
        "dealing with feeling excluded",
        "navigating a boundary issue",
        "after a misunderstanding with a friend",
        "considering ending a relationship",
        "noticing communication patterns",
        "reflecting on a repeated conflict"
        # ... 50+ per domain
    ],
    "career decisions": [
        "after receiving a job offer",
        "considering a career change",
        "processing negative feedback from a boss",
        "deciding whether to speak up in a meeting",
        "evaluating a project failure",
        "thinking about asking for a raise",
        "after being passed over for promotion",
        "considering entrepreneurship",
        "dealing with imposter syndrome",
        "reflecting on work-life balance"
        # ... 50+ per domain
    ],
    # ... for each domain
}
```

### Triggers (what prompts the cognitive action)
```python
triggers = [
    "reading an article that contradicts their view",
    "receiving unexpected feedback",
    "noticing their emotional reaction",
    "having a conversation with someone",
    "experiencing a setback",
    "achieving an unexpected success",
    "seeing someone else's perspective",
    "reflecting during a quiet moment",
    "facing a deadline",
    "encountering similar situation again",
    "being asked a challenging question",
    "overhearing themselves talk about it",
    "writing in a journal",
    "waking up with a new thought",
    "feeling stuck or confused",
    "noticing discomfort with their own position",
    "comparing two different experiences",
    "time passing and gaining distance",
    "reading their old writing or thoughts",
    "discussing it in therapy"
    # ... 100+ triggers
]
```

### Emotional States
```python
emotional_states = [
    "feeling frustrated",
    "experiencing confusion",
    "feeling defensive",
    "in a calm reflective mood",
    "feeling anxious",
    "experiencing curiosity",
    "feeling disappointed",
    "in a moment of clarity",
    "feeling overwhelmed",
    "experiencing relief",
    "feeling resistant",
    "open and receptive",
    "feeling judgmental",
    "experiencing self-doubt",
    "feeling confident",
    "in a vulnerable state",
    "feeling stuck",
    "experiencing hope",
    "feeling skeptical",
    "in a neutral analytical mode"
    # ... 50+ states
]
```

### Language Styles
```python
language_styles = [
    "casual conversational",
    "introspective and literary",
    "straightforward and direct",
    "tentative and exploratory",
    "confident and declarative",
    "stream-of-consciousness",
    "analytical and precise",
    "emotional and expressive",
    "minimalist and spare",
    "detailed and thorough"
]
```

### Unique Angles (force differentiation)
```python
unique_angles = [
    "include a specific sensory detail",
    "show the process taking time (not instant insight)",
    "include self-doubt about the cognitive action itself",
    "show partial or incomplete shift",
    "include resistance before the shift",
    "make it very mundane and everyday",
    "show it happening in a specific physical location",
    "include another person's influence",
    "show it emerging from bodily awareness",
    "frame it as a question rather than statement",
    "include what they're NOT doing (e.g., not blaming)",
    "show mixed feelings about the new perspective",
    "include a metaphor or image",
    "show it happening during an activity (walking, showering, etc.)",
    "include temporal framing (past self vs present self)"
    # ... 50+ angles
]
```

### Forbidden Words (force vocabulary variation)
```python
# Rotate these to prevent repetition
forbidden_word_sets = [
    ["reconsider", "rethink", "think again"],
    ["perspective", "viewpoint", "angle"],
    ["realize", "understand", "see"],
    ["notice", "observe", "aware"],
    ["maybe", "perhaps", "possibly"],
    # ... different sets to rotate through
]
```

### Complexity Levels
```python
complexity_levels = {
    "simple": "Single clear cognitive action, straightforward scenario, obvious outcome",
    "moderate": "Multiple factors at play, some ambiguity, partial clarity",
    "complex": "Multiple interacting cognitive actions, high uncertainty, conflicting considerations, no clear resolution"
}
```

### Perspectives
```python
perspectives = [
    "first-person present tense ('I'm noticing...')",
    "first-person past reflective ('I realized I had been...')",
    "first-person future conditional ('I'll need to...')",
    "second-person coaching ('You might try...')",
    "third-person observation ('She reconsidered...')",
    "internal monologue with self-talk",
    "metacognitive commentary ('My thought process here...')"
]
```

---

## COMBINATION TEMPLATES (For Cognitive Action Chains)

```
TEMPLATE: Chain Prompt
Generate 1 example (3-6 sentences) showing this sequence of cognitive actions:

Step 1: {COGNITIVE_ACTION_1} triggered by {TRIGGER}
Step 2: This leads to {COGNITIVE_ACTION_2}
Step 3: Which results in {COGNITIVE_ACTION_3}

Context:
- Domain: {DOMAIN}
- Scenario: {CONTEXT_DETAIL}
- Subject: {SUBJECT}
- Emotional arc: {STARTING_EMOTION} → {ENDING_EMOTION}

Requirements:
- Show clear progression between steps
- Make causal connections visible
- Complexity: {COMPLEXITY}
- Unique constraint: {UNIQUE_ANGLE}
- Avoid these transitions: {AVOID_TRANSITIONS}

Output only the example text, no preamble.
```

---

## IMPLEMENTATION APPROACH

### Option 1: Pure Random Combination
```python
def generate_unique_prompt(iteration_number):
    template = random.choice(prompt_templates)
    
    params = {
        'COGNITIVE_ACTION': random.choice(cognitive_actions),
        'DOMAIN': random.choice(domains),
        'SUBJECT': random.choice(subjects),
        'TRIGGER': random.choice(triggers),
        'EMOTIONAL_STATE': random.choice(emotional_states),
        'LANGUAGE_STYLE': random.choice(language_styles),
        'UNIQUE_ANGLE': random.choice(unique_angles),
        'COMPLEXITY_LEVEL': random.choice(['simple', 'moderate', 'complex']),
        'PERSPECTIVE_TYPE': random.choice(perspectives),
        'FORBIDDEN_WORDS': random.sample(all_forbidden_words, 3),
        'ITERATION': iteration_number
    }
    
    # Get specific context for domain
    domain = params['DOMAIN']
    params['CONTEXT_DETAIL'] = random.choice(context_details[domain])
    
    # Format the template
    prompt = template.format(**params)
    
    # Add iteration number to ensure uniqueness
    prompt += f"\n\nExample #{iteration_number}. Make this distinctly different from previous examples."
    
    return prompt
```

### Option 2: Stratified Sampling (Better Distribution)
```python
def generate_stratified_prompts(total_examples=100000):
    prompts = []
    
    # Ensure even distribution across cognitive actions
    examples_per_action = total_examples // len(cognitive_actions)
    
    for action in cognitive_actions:
        for i in range(examples_per_action):
            template = random.choice(prompt_templates)
            
            # Force this cognitive action
            params = {
                'COGNITIVE_ACTION': action,
                # Randomize everything else
                'DOMAIN': random.choice(domains),
                'SUBJECT': random.choice(subjects),
                # ... etc
            }
            
            prompt = template.format(**params)
            prompts.append(prompt)
    
    # Shuffle to avoid clustering
    random.shuffle(prompts)
    return prompts
```

### Option 3: Constraint-Based (Ensure Coverage)
```python
# Create a coverage matrix
coverage = {
    action: {
        domain: 0 for domain in domains
    } for action in cognitive_actions
}

def generate_with_coverage_tracking(total_examples):
    prompts = []
    
    for i in range(total_examples):
        # Find under-represented combinations
        min_coverage = min(min(domains.values()) 
                          for domains in coverage.values())
        
        # Select action/domain that need coverage
        candidates = []
        for action, domains_dict in coverage.items():
            for domain, count in domains_dict.items():
                if count == min_coverage:
                    candidates.append((action, domain))
        
        action, domain = random.choice(candidates)
        
        # Generate prompt with these constraints
        prompt = create_prompt(
            cognitive_action=action,
            domain=domain,
            # randomize other parameters
        )
        
        prompts.append(prompt)
        coverage[action][domain] += 1
    
    return prompts
```

---

## COPY-PASTE READY: SINGLE MEGA-PROMPT GENERATOR

Here's a ready-to-use prompt you can copy to an LLM:

```
You are a prompt generator. Generate 1 unique prompt that will be used to create a training example for a cognitive action dataset.

Use this template structure:
"Generate 1 example (2-4 sentences) showing {COGNITIVE_ACTION} in this scenario: {SCENARIO}. Requirements: {CONSTRAINTS}"

Randomly select from these options:

COGNITIVE_ACTIONS: [reconsidering, reframing, noticing patterns, perspective-taking, questioning assumptions, abstracting, concretizing, connecting ideas, distinguishing concepts, updating beliefs, suspending judgment, pattern recognition, zooming out, zooming in, emotional reappraisal, analogical thinking, counterfactual reasoning, hypothesis generation, meta-awareness, accepting]

SUBJECTS: [a developer, a parent, a student, a manager, an artist, a researcher, someone in their 20s, someone in their 60s, a team leader, an entrepreneur, a teacher, a therapist, a writer, a scientist, a career changer, someone grieving, a friend, a partner, someone alone, a person in therapy]

DOMAINS: [relationships, career, creative work, research, ethics, health, finance, education, conflict, identity, parenting, communication, goals, failure, success, daily life, philosophy, team dynamics, time management, personal growth]

TRIGGERS: [feedback received, noticing emotion, conversation, setback, success, seeing others' view, quiet reflection, deadline pressure, repeated pattern, challenging question, journaling, new information, time passing, feeling stuck, physical sensation]

CONTEXTS (pick appropriate for domain): [after difficult conversation, pattern noticed, considering reconnection, processing feedback, feeling excluded, boundary issue, misunderstanding, repeated conflict, job offer received, negative feedback, project failure, considering career change, imposter syndrome, work-life balance]

ANGLES: [include sensory detail, show process taking time, include self-doubt, show partial shift, include resistance, make it mundane, specific location, another person's influence, bodily awareness, frame as question, show what's NOT happening, mixed feelings, include metaphor, during activity, temporal framing]

LANGUAGE_STYLES: [casual, introspective, direct, tentative, confident, stream-of-consciousness, analytical, emotional, minimalist, detailed]

Generate the prompt now with randomly selected values from each category. Number it as #{ITERATION}. Include instruction to avoid repetitive phrases and make it unique.
```

Then you use the OUTPUT of this meta-prompt as the input to generate your actual training examples!

---

