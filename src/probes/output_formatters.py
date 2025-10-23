"""
Output formatting utilities for universal multi-layer inference results
"""
from typing import List, Dict, Optional
from collections import defaultdict


def format_style1_action_grouped(
    predictions: List,
    text: str,
    sentiment_info: Optional[Dict] = None,
    threshold: float = 0.1,
    show_inactive: bool = False
) -> str:
    """
    Format predictions in Style 1: Action-grouped with layer lists and counts

    Example output:
    "So that, in this relationship it's like in your other relationships."
    --------------------------------------------------------------------------------
    Sentiment: Mean=-0.231, Min=-2.591, Max=+0.993
    Predictions grouped by action:
      ✓  1. analogical thinking             (Layers 26, 27, 28, 29, 30)  Count: 24  Max: 1.0000
      ✓  2. self questioning                (Layers 28                )  Count: 15  Max: 1.0000

    Args:
        predictions: List of UniversalPrediction objects
        text: The input text analyzed
        sentiment_info: Optional dict with sentiment statistics {'mean': float, 'min': float, 'max': float}
        threshold: Confidence threshold for filtering
        show_inactive: Whether to show actions below threshold

    Returns:
        Formatted string
    """
    output = []

    # Header with text
    output.append(f'"{text}"')
    output.append("-" * 80)

    # Sentiment info if provided
    if sentiment_info:
        output.append(
            f"Sentiment: Mean={sentiment_info['mean']:+.3f}, "
            f"Min={sentiment_info['min']:+.3f}, "
            f"Max={sentiment_info['max']:+.3f}"
        )

    # Group predictions by action
    action_groups = defaultdict(lambda: {'layers': [], 'confidences': [], 'active_count': 0})

    for pred in predictions:
        if pred.probe_type == "cognitive":  # Only cognitive actions for this format
            action_groups[pred.action_name]['layers'].append(pred.layer)
            action_groups[pred.action_name]['confidences'].append(pred.confidence)
            # Count how many times this action was active (confidence >= threshold)
            if pred.confidence >= threshold:
                action_groups[pred.action_name]['active_count'] += 1

    # Sort by max confidence
    sorted_actions = sorted(
        action_groups.items(),
        key=lambda x: max(x[1]['confidences']),
        reverse=True
    )

    # Filter by threshold if needed
    if not show_inactive:
        sorted_actions = [(name, data) for name, data in sorted_actions
                         if max(data['confidences']) >= threshold]

    # Format output
    output.append("Predictions grouped by action:")

    for idx, (action_name, data) in enumerate(sorted_actions, 1):
        max_conf = max(data['confidences'])
        # Get unique layers where this action was active (>= threshold)
        active_layers = sorted(set([
            layer for layer, conf in zip(data['layers'], data['confidences'])
            if conf >= threshold
        ]))
        count = data['active_count']

        # Format layer list (only show active layers)
        if active_layers:
            layer_str = ", ".join(str(l) for l in active_layers)
        else:
            layer_str = ""

        # Check if active
        marker = "✓" if max_conf >= threshold else " "

        output.append(
            f"  {marker} {idx:2d}. {action_name:30s} "
            f"(Layers {layer_str:20s})  Count: {count:2d}  Max: {max_conf:.4f}"
        )

    return "\n".join(output)


def format_style2_layer_by_layer(
    predictions: List,
    text: str,
    text_index: Optional[int] = None,
    total_texts: Optional[int] = None,
    threshold: float = 0.1,
    min_confidence_display: float = 0.001
) -> str:
    """
    Format predictions in Style 2: Layer-by-layer showing all active actions

    Example output:
    [1/30] Text:
    "The quarterly numbers look... interesting."
    --------------------------------------------------------------------------------
      Layer 21: hypothesis_generation(0.012)
      Layer 22: analyzing(1.000), noticing(1.000), hypothesis_generation(0.996)
      Layer 23: hypothesis_generation(1.000), noticing(1.000), questioning(0.009)

    Args:
        predictions: List of UniversalPrediction objects
        text: The input text analyzed
        text_index: Optional index of current text (1-based)
        total_texts: Optional total number of texts
        threshold: Confidence threshold for filtering
        min_confidence_display: Minimum confidence to display

    Returns:
        Formatted string
    """
    output = []

    # Header with optional index
    if text_index is not None and total_texts is not None:
        output.append(f"[{text_index}/{total_texts}] Text:")
    else:
        output.append("Text:")

    output.append(f'"{text}"')
    output.append("-" * 80)

    # Group predictions by layer
    layer_groups = defaultdict(list)

    for pred in predictions:
        if pred.probe_type == "cognitive":  # Only cognitive actions for this format
            if pred.confidence >= min_confidence_display:
                layer_groups[pred.layer].append((pred.action_name, pred.confidence))

    # Sort layers
    sorted_layers = sorted(layer_groups.keys())

    # Format each layer
    for layer in sorted_layers:
        actions = layer_groups[layer]

        # Sort actions by confidence (descending)
        actions = sorted(actions, key=lambda x: x[1], reverse=True)

        # Format action list
        action_strs = [f"{name}({conf:.3f})" for name, conf in actions]
        action_line = ", ".join(action_strs)

        output.append(f"  Layer {layer}: {action_line}")

    return "\n".join(output)


def extract_sentiment_stats(predictions: List) -> Optional[Dict]:
    """
    Extract sentiment statistics from predictions

    Args:
        predictions: List of UniversalPrediction objects

    Returns:
        Dict with mean, min, max sentiment scores, or None if no sentiment predictions
    """
    sentiment_scores = [
        pred.confidence for pred in predictions
        if pred.probe_type == "sentiment"
    ]

    if not sentiment_scores:
        return None

    return {
        'mean': sum(sentiment_scores) / len(sentiment_scores),
        'min': min(sentiment_scores),
        'max': max(sentiment_scores)
    }


def print_style1_analysis(engine, text: str, threshold: float = 0.1, include_sentiment: bool = True):
    """
    Convenience function to run analysis and print Style 1 format

    Args:
        engine: UniversalMultiLayerInferenceEngine instance
        text: Text to analyze
        threshold: Confidence threshold
        include_sentiment: Whether to include sentiment stats
    """
    # Get predictions
    predictions = engine.predict_all(text, threshold=0.0)  # Get all for processing

    # Extract sentiment if available
    sentiment_info = None
    if include_sentiment:
        sentiment_info = extract_sentiment_stats(predictions)

    # Format and print
    output = format_style1_action_grouped(
        predictions=predictions,
        text=text,
        sentiment_info=sentiment_info,
        threshold=threshold
    )

    print(output)


def print_style2_analysis(
    engine,
    text: str,
    text_index: Optional[int] = None,
    total_texts: Optional[int] = None,
    threshold: float = 0.1,
    min_confidence_display: float = 0.001
):
    """
    Convenience function to run analysis and print Style 2 format

    Args:
        engine: UniversalMultiLayerInferenceEngine instance
        text: Text to analyze
        text_index: Optional index of current text (1-based)
        total_texts: Optional total number of texts
        threshold: Confidence threshold for filtering
        min_confidence_display: Minimum confidence to display
    """
    # Get predictions
    predictions = engine.predict_all(text, threshold=0.0)  # Get all for processing

    # Format and print
    output = format_style2_layer_by_layer(
        predictions=predictions,
        text=text,
        text_index=text_index,
        total_texts=total_texts,
        threshold=threshold,
        min_confidence_display=min_confidence_display
    )

    print(output)


def batch_analyze_style1(engine, texts: List[str], threshold: float = 0.1, include_sentiment: bool = True):
    """
    Analyze multiple texts and print in Style 1 format

    Args:
        engine: UniversalMultiLayerInferenceEngine instance
        texts: List of texts to analyze
        threshold: Confidence threshold
        include_sentiment: Whether to include sentiment stats
    """
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*80}")
        print(f"Analysis {i}/{len(texts)}")
        print('='*80 + "\n")
        print_style1_analysis(engine, text, threshold, include_sentiment)


def batch_analyze_style2(
    engine,
    texts: List[str],
    threshold: float = 0.1,
    min_confidence_display: float = 0.001
):
    """
    Analyze multiple texts and print in Style 2 format

    Args:
        engine: UniversalMultiLayerInferenceEngine instance
        texts: List of texts to analyze
        threshold: Confidence threshold for filtering
        min_confidence_display: Minimum confidence to display
    """
    total = len(texts)
    for i, text in enumerate(texts, 1):
        print_style2_analysis(
            engine,
            text,
            text_index=i,
            total_texts=total,
            threshold=threshold,
            min_confidence_display=min_confidence_display
        )
        print()  # Blank line between analyses