#!/usr/bin/env python3
"""
End-to-End Example: Carl Rogers Therapy Session Analysis

This script demonstrates a complete workflow for analyzing cognitive actions
and sentiment in therapeutic conversations using Brije probe outputs.

What it does:
1. Parses probe-annotated transcripts (cognitive actions + sentiment scores)
2. Aggregates data by speaker (therapist vs client)
3. Generates comparative visualizations showing:
   - Cognitive action frequency comparison
   - Cognitive action bias (therapist-dominant vs client-dominant)
   - Sentiment associations with different cognitive actions

Input: Probe-annotated transcript from interactive TUI output
Output: Publication-quality visualizations (PNG format)

This serves as a template for analyzing any conversation data processed
through the Brije probe system.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

def parse_rogers_session(filepath):
    """Parse the Carl Rogers session file to extract utterances, cognitive actions, and sentiment."""

    with open(filepath, 'r') as f:
        content = f.read()

    # Split by utterance blocks (marked by [N] Speaker:)
    utterance_pattern = r'\[(\d+)\] (.*?):\n"(.*?)"\n-+\nSentiment: Mean=([\+\-]?\d+\.?\d*),.*?\nPredictions grouped by action:\n(.*?)(?=\n\n\[|\Z)'

    matches = re.finditer(utterance_pattern, content, re.DOTALL)

    utterances = []
    for match in matches:
        utt_num = int(match.group(1))
        speaker = match.group(2).strip()
        text = match.group(3)
        sentiment_mean = float(match.group(4))
        actions_block = match.group(5)

        # Parse cognitive actions
        action_pattern = r'✓\s+\d+\.\s+(.*?)\s+\(Layers.*?\)\s+Count:\s+(\d+)'
        actions = {}
        for action_match in re.finditer(action_pattern, actions_block):
            action_name = action_match.group(1).strip()
            count = int(action_match.group(2))
            actions[action_name] = count

        utterances.append({
            'number': utt_num,
            'speaker': speaker,
            'text': text,
            'sentiment': sentiment_mean,
            'actions': actions
        })

    return utterances

def aggregate_by_speaker(utterances):
    """Aggregate cognitive action counts and sentiments by speaker."""

    speaker_data = defaultdict(lambda: {
        'actions': Counter(),
        'sentiments': [],
        'action_sentiments': defaultdict(list)
    })

    for utt in utterances:
        speaker = utt['speaker']
        sentiment = utt['sentiment']

        speaker_data[speaker]['sentiments'].append(sentiment)

        for action, count in utt['actions'].items():
            speaker_data[speaker]['actions'][action] += count
            # Record sentiment for each occurrence of this action
            speaker_data[speaker]['action_sentiments'][action].append(sentiment)

    return speaker_data

def identify_therapist_client(speaker_data):
    """Identify who is the therapist (Carl Rogers) and who is the client (Kathy)."""

    therapist = None
    client = None

    for speaker in speaker_data.keys():
        if 'Carl' in speaker or 'Rogers' in speaker:
            therapist = speaker
        elif 'Kathy' in speaker:
            client = speaker

    return therapist, client

def create_comparison_visualization(speaker_data, therapist, client, output_dir):
    """Create side-by-side comparison visualization similar to AnnoMI analysis."""

    therapist_actions = speaker_data[therapist]['actions']
    client_actions = speaker_data[client]['actions']

    # Get all unique actions
    all_actions = set(therapist_actions.keys()) | set(client_actions.keys())

    # Sort actions by total count (descending)
    action_totals = {action: therapist_actions[action] + client_actions[action]
                     for action in all_actions}
    sorted_actions = sorted(action_totals.keys(), key=lambda x: action_totals[x], reverse=True)

    # Use all actions
    all_actions_sorted = sorted_actions

    # Prepare data
    therapist_counts = [therapist_actions[action] for action in all_actions_sorted]
    client_counts = [client_actions[action] for action in all_actions_sorted]

    # Calculate log2 ratios for bias plot
    log_ratios = []
    for action in all_actions_sorted:
        t_count = therapist_actions[action]
        c_count = client_actions[action]
        if t_count > 0 and c_count > 0:
            log_ratios.append(np.log2(t_count / c_count))
        elif t_count > 0:
            log_ratios.append(10)  # Therapist dominant
        elif c_count > 0:
            log_ratios.append(-10)  # Client dominant
        else:
            log_ratios.append(0)

    # Calculate figure height based on number of actions
    fig_height = max(10, len(all_actions_sorted) * 0.4)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, fig_height))

    # Left plot: Comparison with separate bars
    y_pos = np.arange(len(all_actions_sorted))
    bar_height = 0.35

    ax1.barh(y_pos - bar_height/2, therapist_counts, bar_height, alpha=0.7, label='Therapist (Carl Rogers)', color='#5DADE2')
    ax1.barh(y_pos + bar_height/2, client_counts, bar_height, alpha=0.7, label='Client (Kathy)', color='#F8B88B')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(all_actions_sorted)
    ax1.invert_yaxis()
    ax1.set_xlabel('Aggregate Count', fontsize=12)
    ax1.set_title('Cognitive Action Comparison (Filtered Data):\nTherapist vs Client', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)

    # Right plot: Bias (log2 ratio)
    colors = ['#5DADE2' if r > 0 else '#F8B88B' for r in log_ratios]

    ax2.barh(y_pos, log_ratios, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(all_actions_sorted)
    ax2.invert_yaxis()
    ax2.set_xlabel('Log2(Therapist/Client)', fontsize=12)
    ax2.set_title('Cognitive Action Bias (Filtered Data):\nTherapist-Dominant (blue) vs Client-Dominant (red)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'rogers_kathy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison visualization to {output_path}")
    plt.close()

def create_sentiment_visualization(speaker_data, therapist, client, output_dir):
    """Create visualization showing sentiment associations with cognitive actions."""

    therapist_action_sentiments = speaker_data[therapist]['action_sentiments']
    client_action_sentiments = speaker_data[client]['action_sentiments']

    # Calculate mean sentiment per action for each speaker
    therapist_action_means = {}
    client_action_means = {}

    all_actions = set(therapist_action_sentiments.keys()) | set(client_action_sentiments.keys())

    for action in all_actions:
        if action in therapist_action_sentiments and therapist_action_sentiments[action]:
            therapist_action_means[action] = np.mean(therapist_action_sentiments[action])
        if action in client_action_sentiments and client_action_sentiments[action]:
            client_action_means[action] = np.mean(client_action_sentiments[action])

    # Get actions that appear in both speakers
    common_actions = set(therapist_action_means.keys()) & set(client_action_means.keys())

    # Sort by average sentiment (therapist + client)
    action_avg_sentiments = {
        action: (therapist_action_means.get(action, 0) + client_action_means.get(action, 0)) / 2
        for action in common_actions
    }
    sorted_actions = sorted(action_avg_sentiments.keys(), key=lambda x: action_avg_sentiments[x], reverse=True)

    # Use all common actions
    selected_actions = sorted_actions

    # Calculate figure height based on number of actions
    fig_height = max(10, len(selected_actions) * 0.4)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, fig_height))

    # Left plot: Mean sentiment per action (therapist vs client)
    y_pos = np.arange(len(selected_actions))

    therapist_means = [therapist_action_means.get(action, 0) for action in selected_actions]
    client_means = [client_action_means.get(action, 0) for action in selected_actions]

    width = 0.35
    ax1.barh(y_pos - width/2, therapist_means, width, alpha=0.7, label='Therapist', color='#5DADE2')
    ax1.barh(y_pos + width/2, client_means, width, alpha=0.7, label='Client', color='#F8B88B')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(selected_actions)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean Sentiment Score', fontsize=12)
    ax1.set_title('Cognitive Actions by Mean Sentiment:\nTherapist vs Client', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)

    # Right plot: Sentiment difference (therapist - client)
    sentiment_diff = [t - c for t, c in zip(therapist_means, client_means)]
    colors = ['#5DADE2' if d > 0 else '#F8B88B' for d in sentiment_diff]

    ax2.barh(y_pos, sentiment_diff, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(selected_actions)
    ax2.invert_yaxis()
    ax2.set_xlabel('Sentiment Difference (Therapist - Client)', fontsize=12)
    ax2.set_title('Cognitive Action Sentiment Bias:\nTherapist More Positive (blue) vs Client More Positive (red)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'rogers_kathy_sentiment_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sentiment visualization to {output_path}")
    plt.close()

def print_summary_stats(speaker_data, therapist, client):
    """Print summary statistics."""

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for speaker in [therapist, client]:
        print(f"\n{speaker}:")
        print(f"  Total utterances: {len(speaker_data[speaker]['sentiments'])}")
        print(f"  Total cognitive actions: {sum(speaker_data[speaker]['actions'].values())}")
        print(f"  Unique cognitive actions: {len(speaker_data[speaker]['actions'])}")
        print(f"  Mean sentiment: {np.mean(speaker_data[speaker]['sentiments']):.3f}")
        print(f"  Sentiment range: [{min(speaker_data[speaker]['sentiments']):.3f}, {max(speaker_data[speaker]['sentiments']):.3f}]")

        # Top 5 actions
        top_5 = speaker_data[speaker]['actions'].most_common(5)
        print(f"  Top 5 actions:")
        for action, count in top_5:
            print(f"    - {action}: {count}")

def main():
    # File paths (relative to Keep_viz directory)
    input_file = Path("data/Kathy_session_interactive_analysis.txt")
    output_dir = Path(".")  # Save visualizations in Keep_viz directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing Carl Rogers session data...")
    utterances = parse_rogers_session(input_file)
    print(f"Parsed {len(utterances)} utterances")

    print("\nAggregating data by speaker...")
    speaker_data = aggregate_by_speaker(utterances)

    print("\nIdentifying therapist and client...")
    therapist, client = identify_therapist_client(speaker_data)
    print(f"Therapist: {therapist}")
    print(f"Client: {client}")

    print_summary_stats(speaker_data, therapist, client)

    print("\nCreating comparison visualization...")
    create_comparison_visualization(speaker_data, therapist, client, output_dir)

    print("\nCreating sentiment visualization...")
    create_sentiment_visualization(speaker_data, therapist, client, output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()