"""
Variable Pools for Cognitive Action Training Data Generation
Based on scientific taxonomies from cognitive psychology and emotion research
"""

import random

# =============================================================================
# COGNITIVE ACTIONS (Based on Scientific Taxonomies)
# =============================================================================

# Combined from multiple taxonomies: original instructions + scientific frameworks
COGNITIVE_ACTIONS = {
    # Original Core Actions
    "reconsidering": "reconsidering a belief or decision",
    "reframing": "reframing a situation or perspective",
    "noticing": "noticing a pattern, feeling, or dynamic",
    "perspective_taking": "taking another's perspective or temporal view",
    "questioning": "questioning an assumption or belief",
    "abstracting": "abstracting from specifics to general patterns",
    "concretizing": "making abstract concepts concrete and specific",
    "connecting": "connecting disparate ideas or experiences",
    "distinguishing": "distinguishing between previously conflated concepts",
    "updating_beliefs": "updating mental models or beliefs",
    "suspending_judgment": "suspending judgment and staying with uncertainty",
    "pattern_recognition": "recognizing recurring patterns across situations",
    "zooming_out": "zooming out for broader context",
    "zooming_in": "zooming in on specific details",
    "analogical_thinking": "drawing analogies between domains",
    "counterfactual_reasoning": "engaging in 'what if' thinking",
    "hypothesis_generation": "generating possible explanations",
    "meta_awareness": "reflecting on one's own thinking process",
    "accepting": "accepting and letting go of control",

    # From Bloom's Taxonomy
    "remembering": "recalling relevant information or experiences",
    "understanding": "interpreting and explaining meaning",
    "applying": "using knowledge in new situations",
    "analyzing": "breaking down into components",
    "evaluating": "making judgments about value or effectiveness",
    "creating": "generating new ideas or solutions",

    # From Guilford's Structure of Intellect
    "divergent_thinking": "generating multiple creative solutions",
    "convergent_thinking": "finding the single best solution",
    "cognition_awareness": "becoming aware and comprehending",

    # Metacognitive Operations
    "metacognitive_monitoring": "tracking one's own comprehension",
    "metacognitive_regulation": "adjusting thinking strategies",
    "self_questioning": "interrogating one's own understanding",

    # Emotional/Affective Operations (from taxonomies)
    "emotional_reappraisal": "reinterpreting emotional meaning",
    "emotion_receiving": "becoming aware of emotions",
    "emotion_responding": "actively engaging with emotions",
    "emotion_valuing": "attaching worth to emotional experiences",
    "emotion_organizing": "integrating conflicting emotions",
    "emotion_characterizing": "aligning emotions with core values",
    "situation_selection": "choosing emotional contexts deliberately",
    "situation_modification": "changing circumstances to regulate emotion",
    "attentional_deployment": "directing attention for emotional regulation",
    "response_modulation": "modifying emotional expression",
    "emotion_perception": "identifying emotions in self/others",
    "emotion_facilitation": "using emotions to enhance thinking",
    "emotion_understanding": "comprehending emotional complexity",
    "emotion_management": "regulating emotions in self/others"
}

# =============================================================================
# SUBJECTS (Who is doing the thinking)
# =============================================================================

SUBJECTS = [
    # Professional Roles
    "a software developer", "a teacher", "a therapist", "a manager", "a researcher",
    "a scientist", "a doctor", "a lawyer", "a consultant", "a designer",
    "an engineer", "a writer", "an artist", "a musician", "an entrepreneur",
    "a nurse", "a social worker", "a coach", "a mentor", "a leader",

    # Life Stages/Demographics
    "someone in their early 20s", "someone in their 30s", "someone in their 40s",
    "someone in their 50s", "someone in their 60s", "a recent graduate",
    "a parent", "a grandparent", "a student", "a retiree",

    # Relationship Roles
    "a partner in a relationship", "a friend", "a colleague", "a team member",
    "a sibling", "a child reflecting on parents", "a mentor", "a mentee",

    # Life Situations
    "someone grieving a loss", "someone facing a major transition",
    "someone dealing with success", "someone processing failure",
    "a person in therapy", "someone in recovery", "a career changer",
    "someone learning a new skill", "a person facing illness",
    "someone in conflict", "a person seeking growth"
]

# =============================================================================
# DOMAINS (Context areas)
# =============================================================================

DOMAINS = [
    "personal relationships", "romantic relationships", "family dynamics",
    "friendships", "career decisions", "professional development",
    "creative work", "artistic expression", "scientific research",
    "academic learning", "moral and ethical dilemmas", "health and wellness",
    "financial planning", "investment decisions", "conflict resolution",
    "identity and self-concept", "parenting and caregiving", "leadership challenges",
    "team dynamics", "communication challenges", "goal setting and achievement",
    "dealing with failure", "processing success", "daily mundane decisions",
    "philosophical questions", "spiritual exploration", "time management",
    "personal growth", "therapy and healing", "addiction recovery",
    "grief and loss", "major life transitions", "retirement planning",
    "educational choices", "political beliefs", "social justice issues"
]

# =============================================================================
# CONTEXT DETAILS (Specific scenarios for each domain)
# =============================================================================

CONTEXT_DETAILS = {
    "personal relationships": [
        "after a difficult conversation with a partner",
        "noticing a recurring conflict pattern with a family member",
        "considering whether to reconnect with an old friend",
        "processing feedback from someone close",
        "dealing with feeling excluded from a social group",
        "navigating a boundary issue with a friend",
        "after a misunderstanding was clarified",
        "considering ending a toxic relationship",
        "noticing how they communicate differently with different people",
        "reflecting on a repeated argument pattern"
    ],

    "career decisions": [
        "after receiving a job offer from a different field",
        "considering a major career pivot",
        "processing harsh feedback from a supervisor",
        "deciding whether to speak up about workplace issues",
        "evaluating why a project failed",
        "thinking about asking for a promotion",
        "after being passed over for advancement",
        "considering starting their own business",
        "dealing with imposter syndrome at work",
        "reflecting on work-life balance priorities"
    ],

    "creative work": [
        "after receiving criticism of their artistic work",
        "experiencing creative block for weeks",
        "comparing their work to others in their field",
        "deciding whether to pursue commercial vs. artistic goals",
        "processing rejection from publishers/galleries",
        "considering a new creative direction",
        "dealing with creative self-doubt",
        "balancing creative passion with financial needs",
        "after receiving unexpected praise for their work",
        "questioning their artistic vision"
    ],

    "health and wellness": [
        "after receiving concerning medical news",
        "trying to understand persistent symptoms",
        "deciding whether to start a major lifestyle change",
        "processing a diagnosis that changes everything",
        "considering alternative treatment approaches",
        "dealing with chronic pain or illness",
        "reflecting on eating and exercise patterns",
        "processing addiction or recovery experiences",
        "considering mental health treatment options",
        "balancing multiple health priorities"
    ],

    "family dynamics": [
        "processing childhood experiences affecting present relationships",
        "dealing with aging parents' changing needs",
        "navigating conflicts between family members",
        "considering how family patterns affect current behavior",
        "processing inherited trauma or family history",
        "dealing with family expectations vs. personal desires",
        "navigating holiday gatherings and family traditions",
        "processing the death of a family member",
        "considering how to break generational cycles",
        "dealing with family financial stress"
    ]
}

# Add more context details for other domains
for domain in DOMAINS:
    if domain not in CONTEXT_DETAILS:
        CONTEXT_DETAILS[domain] = [
            f"facing a significant decision about {domain}",
            f"processing unexpected developments in {domain}",
            f"reflecting on patterns in {domain}",
            f"considering changes to their approach in {domain}",
            f"dealing with conflict or tension in {domain}"
        ]

# =============================================================================
# TRIGGERS (What prompts the cognitive action)
# =============================================================================

TRIGGERS = [
    "reading an article that contradicts their worldview",
    "receiving unexpected feedback from someone they trust",
    "noticing their physical or emotional reaction to something",
    "having a meaningful conversation with someone",
    "experiencing an unexpected setback or failure",
    "achieving success in an unexpected way",
    "witnessing someone else's perspective on the same issue",
    "having quiet time for reflection during a walk or shower",
    "facing a deadline that forces clarity",
    "encountering a similar situation to one from their past",
    "being asked a challenging question they couldn't answer",
    "overhearing themselves explain their position to someone",
    "writing in a journal or diary",
    "waking up with a new thought after sleeping on it",
    "feeling stuck or confused about a decision",
    "noticing discomfort with their own stated position",
    "comparing two different experiences or approaches",
    "time passing and gaining emotional distance",
    "re-reading their old writing or notes",
    "discussing the issue in therapy or with a counselor",
    "observing their pattern across multiple situations",
    "receiving new information that doesn't fit their model",
    "noticing someone else struggling with similar issues",
    "experiencing a moment of unexpected clarity",
    "being challenged by someone they respect"
]

# =============================================================================
# EMOTIONAL STATES
# =============================================================================

EMOTIONAL_STATES = [
    "feeling frustrated and stuck", "experiencing confusion and uncertainty",
    "feeling defensive about their position", "in a calm and reflective mood",
    "feeling anxious about the implications", "experiencing genuine curiosity",
    "feeling disappointed by outcomes", "in a moment of unexpected clarity",
    "feeling overwhelmed by options", "experiencing relief after stress",
    "feeling resistant to change", "open and receptive to new ideas",
    "feeling judgmental toward others", "experiencing self-doubt",
    "feeling confident in their abilities", "in a vulnerable emotional state",
    "feeling intellectually stuck", "experiencing hope about possibilities",
    "feeling skeptical of new information", "in a neutral analytical mindset",
    "feeling emotionally drained", "experiencing excitement about discovery",
    "feeling protective of their beliefs", "in a state of creative flow",
    "feeling pressured by circumstances", "experiencing gratitude for insights"
]

# =============================================================================
# LANGUAGE STYLES
# =============================================================================

LANGUAGE_STYLES = [
    "casual and conversational",
    "introspective and literary",
    "straightforward and direct",
    "tentative and exploratory",
    "confident and declarative",
    "stream-of-consciousness style",
    "analytical and precise",
    "emotional and expressive",
    "minimalist and spare",
    "detailed and thorough",
    "questioning and uncertain",
    "philosophical and reflective"
]

# =============================================================================
# UNIQUE ANGLES (Force differentiation)
# =============================================================================

UNIQUE_ANGLES = [
    "include a specific sensory detail that triggered the insight",
    "show the cognitive process taking time rather than being instant",
    "include self-doubt about the cognitive process itself",
    "show a partial or incomplete cognitive shift",
    "include resistance or pushback before the mental shift",
    "make the scenario very mundane and everyday",
    "show it happening in a specific physical location",
    "include another person's influence on the thinking",
    "show it emerging from bodily awareness or sensation",
    "frame the insight as a question rather than a statement",
    "include what they're explicitly NOT doing (e.g., not blaming)",
    "show mixed or conflicted feelings about the new perspective",
    "include a specific metaphor or mental image",
    "show it happening during a routine activity",
    "include temporal framing (past self vs. present self)",
    "show the cognitive action being interrupted or incomplete",
    "include uncertainty about whether the new perspective is right",
    "show multiple cognitive actions happening simultaneously",
    "include how the insight affects their body or energy",
    "show the cognitive action being triggered by memory"
]

# =============================================================================
# COMPLEXITY LEVELS
# =============================================================================

COMPLEXITY_LEVELS = {
    "simple": "Single clear cognitive action, straightforward scenario, obvious outcome",
    "moderate": "Multiple factors at play, some ambiguity, partial clarity",
    "complex": "Multiple interacting cognitive actions, high uncertainty, conflicting considerations, no clear resolution"
}

# =============================================================================
# PERSPECTIVES
# =============================================================================

PERSPECTIVES = [
    "first-person present tense ('I'm noticing right now...')",
    "first-person past reflective ('I realized later that I had been...')",
    "first-person future conditional ('I'll need to reconsider when...')",
    "second-person coaching ('You might try reframing...')",
    "third-person observation ('She began to reconsider...')",
    "internal monologue with self-talk",
    "metacognitive commentary ('My thought process here is...')"
]

# =============================================================================
# FORBIDDEN WORDS (Rotate to prevent repetition)
# =============================================================================

FORBIDDEN_WORD_SETS = [
    ["reconsider", "rethink", "think again"],
    ["perspective", "viewpoint", "angle", "lens"],
    ["realize", "understand", "see", "get it"],
    ["notice", "observe", "become aware", "see"],
    ["maybe", "perhaps", "possibly", "might"],
    ["reframe", "see differently", "look at it as"],
    ["pattern", "trend", "recurring theme"],
    ["assumption", "belief", "given", "take for granted"],
    ["insight", "aha moment", "breakthrough", "epiphany"],
    ["shift", "change", "move", "transition"]
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_random_selection():
    """Get a random selection of variables for prompt generation"""
    return {
        'cognitive_action': random.choice(list(COGNITIVE_ACTIONS.keys())),
        'cognitive_action_description': lambda action: COGNITIVE_ACTIONS[action],
        'subject': random.choice(SUBJECTS),
        'domain': random.choice(DOMAINS),
        'trigger': random.choice(TRIGGERS),
        'emotional_state': random.choice(EMOTIONAL_STATES),
        'language_style': random.choice(LANGUAGE_STYLES),
        'unique_angle': random.choice(UNIQUE_ANGLES),
        'complexity_level': random.choice(list(COMPLEXITY_LEVELS.keys())),
        'perspective': random.choice(PERSPECTIVES),
        'forbidden_words': []  # Will be properly set in template generation
    }

def get_context_detail(domain):
    """Get a specific context detail for a domain"""
    return random.choice(CONTEXT_DETAILS.get(domain, CONTEXT_DETAILS["personal relationships"]))

def get_cognitive_action_chains(n=3):
    """Get a sequence of cognitive actions for chain examples"""
    actions = random.sample(list(COGNITIVE_ACTIONS.keys()), n)
    return actions