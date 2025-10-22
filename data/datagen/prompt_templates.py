"""
Prompt Templates for Cognitive Action Training Data Generation
Based on instructions2.md template system architecture
"""

import random
from variable_pools import *

# =============================================================================
# SINGLE COGNITIVE ACTION TEMPLATES
# =============================================================================

SINGLE_ACTION_TEMPLATES = [
    # Template 1: Scenario-Based
    """Generate 1 example (2-4 sentences) showing someone {cognitive_action_desc} in this specific scenario:

Scenario: {subject} is {situation} and experiences {trigger}. They engage in {cognitive_action_desc}.

Requirements:
- Domain: {domain}
- Emotional context: {emotional_state}
- Show the cognitive process explicitly
- Use {language_style} language
- Focus angle: {unique_angle}
- Do NOT use these words: {forbidden_words}
- Complexity: {complexity_level}

Output only the example text, no preamble.""",

    # Template 2: Before/After
    """Generate 1 example (2-4 sentences) showing {cognitive_action_desc} by contrasting before and after states:

Context: {domain} situation involving {context_detail}
Before state: {initial_state}
Cognitive action: {cognitive_action_desc}
After state: {result_direction}

Requirements:
- Subject: {subject}
- Complexity: {complexity_level}
- Include specific details about {focus_element}
- Avoid cliché phrasings
- Language register: {language_style}
- Unique constraint: {unique_angle}

Output only the example text, no preamble.""",

    # Template 3: Process-Focused
    """Generate 1 example (2-4 sentences) that shows the PROCESS of {cognitive_action_desc}:

Setup: {subject} faces {problem_type} related to {domain}
The cognitive action unfolds through these stages: {process_stages}

Requirements:
- Make the internal process visible
- Include: {required_element}
- Perspective: {perspective}
- Incorporate this complication: {complication}
- Avoid these phrases: {avoid_phrases}
- Unique angle: {unique_angle}

Output only the example text, no preamble.""",

    # Template 4: Trigger-Response
    """Generate 1 example (2-4 sentences) where {trigger} leads to {cognitive_action_desc}:

Context: {context_detail} in {domain}
Trigger: {specific_trigger}
Cognitive response: {cognitive_action_desc}
Outcome direction: {outcome_direction}

Requirements:
- Subject type: {subject}
- Time frame: {time_frame}
- Include sensory or emotional detail
- Unique constraint: {unique_angle}
- Language style: {language_style}
- Avoid repetitive phrasing

Output only the example text, no preamble.""",

    # Template 5: Meta-Cognitive Focus
    """Generate 1 example (2-4 sentences) showing {cognitive_action_desc} with explicit meta-awareness:

Context: {subject} is {situation} in {domain}
Meta-cognitive element: They become aware of their own {metacognitive_focus}
Cognitive action: {cognitive_action_desc}

Requirements:
- Show awareness OF the thinking process
- Emotional context: {emotional_state}
- Include: {metacognitive_element}
- Perspective: {perspective}
- Unique angle: {unique_angle}
- Complexity: {complexity_level}

Output only the example text, no preamble."""
]

# =============================================================================
# COGNITIVE ACTION CHAIN TEMPLATES
# =============================================================================

CHAIN_TEMPLATES = [
    # Template: Sequential Chain
    """Generate 1 example (4-6 sentences) showing this sequence of cognitive actions:

Step 1: {cognitive_action_1_desc} triggered by {trigger}
Step 2: This leads to {cognitive_action_2_desc}
Step 3: Which results in {cognitive_action_3_desc}

Context:
- Domain: {domain}
- Scenario: {context_detail}
- Subject: {subject}
- Emotional arc: {starting_emotion} → {ending_emotion}

Requirements:
- Show clear progression between steps
- Make causal connections visible
- Complexity: {complexity}
- Unique constraint: {unique_angle}
- Avoid these transitions: {avoid_transitions}

Output only the example text, no preamble.""",

    # Template: Parallel Processing
    """Generate 1 example (3-5 sentences) showing multiple cognitive actions happening simultaneously:

Cognitive actions occurring together:
- {cognitive_action_1_desc}
- {cognitive_action_2_desc}
- {cognitive_action_3_desc}

Context: {subject} is {situation} in {domain}
Trigger: {trigger}

Requirements:
- Show actions as simultaneous, not sequential
- Include internal tension or complexity
- Emotional state: {emotional_state}
- Unique angle: {unique_angle}
- Language style: {language_style}

Output only the example text, no preamble."""
]

# =============================================================================
# DIALOGUE TEMPLATES
# =============================================================================

DIALOGUE_TEMPLATES = [
    # Therapy/Coaching Context
    """Generate a dialogue (3-4 exchanges) showing {cognitive_action_desc} in a therapeutic context:

Setting: {therapy_setting}
Client issue: {domain} - {context_detail}
Cognitive action demonstrated: {cognitive_action_desc}

Requirements:
- Show the cognitive action emerging through dialogue
- Include both therapist and client voices
- Emotional tone: {emotional_state}
- Make it feel natural and realistic
- Unique focus: {unique_angle}

Output dialogue format only, no stage directions.""",

    # Peer Conversation
    """Generate a dialogue (2-3 exchanges) between peers showing {cognitive_action_desc}:

Setting: {conversation_setting}
Topic: {domain} - {context_detail}
Participants: {subject} and a friend/colleague

Requirements:
- Show cognitive action emerging through conversation
- Keep dialogue natural and realistic
- Include: {unique_angle}
- Language style: {language_style}
- Emotional undertone: {emotional_state}

Output dialogue format only."""
]

# =============================================================================
# THOUGHT-STREAM TEMPLATES
# =============================================================================

THOUGHT_STREAM_TEMPLATES = [
    # Stream of Consciousness
    """Generate a stream-of-consciousness example (4-6 sentences) showing {cognitive_action_desc}:

Initial state: {subject} is {initial_situation} feeling {emotional_state}
Domain: {domain}
Trigger: {trigger}
Cognitive process: {cognitive_action_desc}

Requirements:
- Show the mind in motion, not just the conclusion
- Include false starts, interruptions, tangents
- Make it feel like real internal dialogue
- Include: {unique_angle}
- Complexity: {complexity_level}
- Length: {length} words approximately

Output only the thought stream, no framing.""",

    # Internal Negotiation
    """Generate an internal dialogue (4-5 sentences) showing {cognitive_action_desc} as negotiation between different parts of mind:

Context: {subject} facing {context_detail} in {domain}
Internal voices: {internal_voices}
Cognitive action: {cognitive_action_desc}

Requirements:
- Show different perspectives within one person
- Include emotional complexity
- Unique angle: {unique_angle}
- Resolution style: {resolution_style}
- Language: {language_style}

Output only the internal dialogue."""
]

# =============================================================================
# NEGATIVE EXAMPLE TEMPLATES
# =============================================================================

NEGATIVE_TEMPLATES = [
    """Generate 1 example (2-4 sentences) showing the ABSENCE of {cognitive_action_desc}:

Context: {subject} faces {context_detail} in {domain}
Trigger present: {trigger}
What they DON'T do: {cognitive_action_desc}
Instead they: {negative_alternative}

Requirements:
- Show rigid thinking or missed opportunity
- Make it realistic (not caricature)
- Emotional state: {emotional_state}
- Include: {unique_angle}

Output only the example text, no preamble."""
]

# =============================================================================
# TEMPLATE PARAMETER GENERATORS
# =============================================================================

def generate_template_parameters(cognitive_action_key, template_type="single"):
    """Generate all parameters needed for template formatting"""
    base_params = get_random_selection()
    domain = base_params['domain']

    params = {
        'cognitive_action': cognitive_action_key,
        'cognitive_action_desc': COGNITIVE_ACTIONS[cognitive_action_key],
        'subject': base_params['subject'],
        'domain': domain,
        'context_detail': get_context_detail(domain),
        'trigger': base_params['trigger'],
        'emotional_state': base_params['emotional_state'],
        'language_style': base_params['language_style'],
        'unique_angle': base_params['unique_angle'],
        'complexity_level': base_params['complexity_level'],
        'perspective': base_params['perspective'],
        'forbidden_words': ', '.join(base_params['forbidden_words']),
    }

    # Add common template parameters used across all types
    params.update({
        'situation': f"dealing with {get_context_detail(domain)}",
        'initial_state': f"initially feeling {random.choice(EMOTIONAL_STATES)}",
        'result_direction': f"moving toward {random.choice(['clarity', 'acceptance', 'understanding', 'resolution', 'peace'])}",
        'focus_element': random.choice(['emotional reactions', 'underlying assumptions', 'behavioral patterns', 'decision criteria']),
        'problem_type': f"a challenge involving {random.choice(['conflicting values', 'unclear options', 'emotional complexity', 'competing priorities'])}",
        'process_stages': "awareness → questioning → exploration → insight",
        'required_element': random.choice(['self-compassion', 'specific details', 'emotional honesty', 'multiple perspectives']),
        'complication': random.choice(['time pressure', 'conflicting advice', 'emotional resistance', 'unclear information']),
        'avoid_phrases': ', '.join(random.sample(['I realized', 'suddenly understood', 'it hit me', 'I saw that'], 2)),
        'specific_trigger': params['trigger'],
        'outcome_direction': random.choice(['greater clarity', 'reduced anxiety', 'new perspective', 'deeper understanding']),
        'time_frame': random.choice(['in that moment', 'over several days', 'gradually', 'after some time']),
        'metacognitive_focus': random.choice(['thinking patterns', 'emotional reactions', 'decision process', 'assumptions']),
        'metacognitive_element': random.choice(['awareness of their own bias', 'recognition of their pattern', 'questioning their approach'])
    })

    # Add template-specific parameters
    if template_type == "single":
        pass  # All parameters already added above

    elif template_type == "chain":
        actions = get_cognitive_action_chains(3)
        params.update({
            'cognitive_action_1': actions[0],
            'cognitive_action_1_desc': COGNITIVE_ACTIONS[actions[0]],
            'cognitive_action_2': actions[1],
            'cognitive_action_2_desc': COGNITIVE_ACTIONS[actions[1]],
            'cognitive_action_3': actions[2],
            'cognitive_action_3_desc': COGNITIVE_ACTIONS[actions[2]],
            'starting_emotion': random.choice(EMOTIONAL_STATES),
            'ending_emotion': random.choice(EMOTIONAL_STATES),
            'complexity': params['complexity_level'],
            'avoid_transitions': ', '.join(['then I', 'next I', 'after that']),
            'situation': f"dealing with {get_context_detail(domain)}"
        })

    elif template_type == "dialogue":
        params.update({
            'therapy_setting': random.choice(['therapy session', 'coaching conversation', 'peer support group']),
            'conversation_setting': random.choice(['coffee shop', 'walking together', 'phone call', 'lunch meeting'])
        })

    elif template_type == "thought_stream":
        params.update({
            'initial_situation': f"dealing with {get_context_detail(domain)}",
            'length': random.choice([100, 150, 200, 250]),
            'internal_voices': random.choice(['practical vs. emotional', 'cautious vs. optimistic', 'critical vs. compassionate']),
            'resolution_style': random.choice(['tentative conclusion', 'ongoing tension', 'partial clarity', 'new questions'])
        })

    elif template_type == "negative":
        params.update({
            'negative_alternative': random.choice([
                'stick rigidly to their original view',
                'dismiss the new information',
                'avoid thinking about it',
                'become more defensive',
                'blame external circumstances'
            ])
        })

    return params

# =============================================================================
# MAIN TEMPLATE SELECTION FUNCTIONS
# =============================================================================

def get_random_template(template_type="single"):
    """Get a random template of specified type"""
    if template_type == "single":
        return random.choice(SINGLE_ACTION_TEMPLATES)
    elif template_type == "chain":
        return random.choice(CHAIN_TEMPLATES)
    elif template_type == "dialogue":
        return random.choice(DIALOGUE_TEMPLATES)
    elif template_type == "thought_stream":
        return random.choice(THOUGHT_STREAM_TEMPLATES)
    elif template_type == "negative":
        return random.choice(NEGATIVE_TEMPLATES)
    else:
        return random.choice(SINGLE_ACTION_TEMPLATES)

def generate_prompt(cognitive_action_key=None, cognitive_action=None, template_type="single", iteration_number=0):
    """Generate a complete prompt for the LLM"""
    # Handle both parameter names for backwards compatibility
    if cognitive_action_key is None and cognitive_action is not None:
        cognitive_action_key = cognitive_action
    elif cognitive_action_key is None:
        cognitive_action_key = random.choice(list(COGNITIVE_ACTIONS.keys()))

    template = get_random_template(template_type)
    params = generate_template_parameters(cognitive_action_key, template_type)

    # Format the template with parameters
    try:
        prompt = template.format(**params)
    except KeyError as e:
        # Fallback: add missing parameters
        missing_param = str(e).strip("'")
        print(f"Warning: Missing parameter '{missing_param}' for template type '{template_type}'")
        params[missing_param] = f"[{missing_param}]"
        prompt = template.format(**params)

    # Add uniqueness constraint
    prompt += f"\n\nExample #{iteration_number + 1}. Make this distinctly different from previous examples."

    return prompt, params

def generate_batch_prompts(batch_size=10, cognitive_action=None, template_type="single"):
    """Generate a batch of prompts for the same cognitive action"""
    prompts = []
    for i in range(batch_size):
        prompt, params = generate_prompt(cognitive_action, template_type, i)
        prompts.append({
            'prompt': prompt,
            'cognitive_action': params['cognitive_action'],
            'domain': params['domain'],
            'template_type': template_type,
            'iteration': i
        })
    return prompts