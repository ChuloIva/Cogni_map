"""
Canonical cognitive action categories and helpers.

Categories (5):
 - metacognitive
 - analytical
 - creative
 - affective
 - memory

Mapping is built against action keys from third_party/datagen/variable_pools.COGNITIVE_ACTIONS.
"""

from typing import Dict


# Short tags for compact labels
CATEGORY_TAGS: Dict[str, str] = {
    "metacognitive": "META",
    "analytical": "ANL",
    "creative": "CRV",
    "emotional": "EMO",
    "memory": "MEM",
    "other": "OTH",
}


# Canonical mapping from action name -> category
ACTION_TO_CATEGORY: Dict[str, str] = {
    # Metacognitive
    "reconsidering": "metacognitive",
    "updating_beliefs": "metacognitive",
    "suspending_judgment": "metacognitive",
    "meta_awareness": "metacognitive",
    "metacognitive_monitoring": "metacognitive",
    "metacognitive_regulation": "metacognitive",
    "self_questioning": "metacognitive",

    # Analytical / Reasoning / Attention
    "noticing": "analytical",
    "pattern_recognition": "analytical",
    "zooming_out": "analytical",
    "zooming_in": "analytical",
    "questioning": "analytical",
    "abstracting": "analytical",
    "concretizing": "analytical",
    "connecting": "analytical",
    "distinguishing": "analytical",
    "perspective_taking": "analytical",
    "convergent_thinking": "analytical",
    "understanding": "analytical",
    "applying": "analytical",
    "analyzing": "analytical",
    "evaluating": "analytical",
    "cognition_awareness": "analytical",

    # Creative / Divergent
    "creating": "creative",
    "divergent_thinking": "creative",
    "hypothesis_generation": "creative",
    "counterfactual_reasoning": "creative",
    "analogical_thinking": "creative",
    "reframing": "creative",

    # Emotional / Affective regulation (Gross, etc.)
    "emotional_reappraisal": "emotional",
    "emotion_receiving": "emotional",
    "emotion_responding": "emotional",
    "emotion_valuing": "emotional",
    "emotion_organizing": "emotional",
    "emotion_characterizing": "emotional",
    "situation_selection": "emotional",
    "situation_modification": "emotional",
    "attentional_deployment": "emotional",
    "response_modulation": "emotional",
    "emotion_perception": "emotional",
    "emotion_facilitation": "emotional",
    "emotion_understanding": "emotional",
    "emotion_management": "emotional",
    "accepting": "emotional",

    # Memory (Bloom: Remember)
    "remembering": "memory",
}


def get_action_category(action_name: str) -> str:
    """Return canonical category for a given action name."""
    key = (action_name or "").strip()
    return ACTION_TO_CATEGORY.get(key, "other")


def get_category_tag(category: str) -> str:
    """Return short tag for a category."""
    return CATEGORY_TAGS.get(category, CATEGORY_TAGS["other"])


