"""
Interactive Terminal UI for Probe Visualization

Full-screen TUI with navigable panels using Textual framework
"""

from .gpu_utils import configure_amd_gpu
configure_amd_gpu()

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import asyncio

# Add paths
NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, DataTable, Label
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from rich.table import Table as RichTable
from rich.panel import Panel

from streaming_probe_inference import StreamingProbeInferenceEngine, StreamingPrediction, AggregatedPrediction
from visualization_utils import (
    confidence_to_color,
    format_confidence,
    truncate_text,
    create_sparkline,
    get_emoji_for_confidence,
    format_timestamp,
    get_activation_symbol,
    get_category_color
)
from action_categories import get_action_category, get_category_tag


class TokenDisplay(ScrollableContainer):
    """Display tokens with highlighting for selected token - SCROLLABLE VERSION"""

    selected_token = reactive(0)

    def __init__(self, tokens: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = tokens
        self.predictions: Optional[List[StreamingPrediction]] = None
        self.display_threshold: float = 0.1
        self.token_static = Static()

    def compose(self) -> ComposeResult:
        """Compose the scrollable container"""
        yield self.token_static

    def set_predictions(self, predictions: List[StreamingPrediction]):
        """Set predictions data"""
        self.predictions = predictions
        self.refresh_display()

    def refresh_display(self):
        """Refresh the token display"""
        text = Text()

        # Wrap tokens to fit nicely - aim for ~100 chars per line
        current_line_length = 0
        max_line_length = 100

        for i, token in enumerate(self.tokens):
            # Check if this token has activations
            has_activation = False
            if self.predictions:
                for pred in self.predictions[:5]:  # Top 5
                    # Handle both AggregatedPrediction and StreamingPrediction
                    token_acts = pred.token_activations if hasattr(pred, 'token_activations') else (pred.layer_predictions[0].token_activations if pred.layer_predictions else [])
                    if i < len(token_acts):
                        if token_acts[i].confidence >= self.display_threshold:
                            has_activation = True
                            break

            # Add newline if line is getting too long
            token_display_length = len(token) + 2  # +2 for brackets or space
            if current_line_length + token_display_length > max_line_length and current_line_length > 0:
                text.append("\n")
                current_line_length = 0

            # Style based on state
            if i == self.selected_token:
                # Selected: bold and highlighted
                text.append(f"[{token}]", style="bold black on yellow")
                current_line_length += len(token) + 2
            elif has_activation:
                # Has activation: colored
                text.append(token, style="bold cyan")
                current_line_length += len(token)
            else:
                # Normal
                text.append(token, style="dim white")
                current_line_length += len(token)

            text.append(" ")
            current_line_length += 1

        self.token_static.update(Panel(text, title="Token Stream (←/→ to navigate, scroll with mouse/trackpad)", border_style="blue"))

    def watch_selected_token(self, new_value: int) -> None:
        """Called when selected_token changes"""
        self.refresh_display()


class ProbeDetails(Static):
    """Display details for selected token"""

    selected_token = reactive(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictions: Optional[List[AggregatedPrediction]] = None
        self.tokens: List[str] = []
        self.display_threshold: float = 0.1

    def set_data(self, predictions: List[AggregatedPrediction], tokens: List[str]):
        """Set data"""
        self.predictions = predictions
        self.tokens = tokens
        self.refresh_details()

    def refresh_details(self):
        """Refresh the details display"""
        if not self.predictions or not self.tokens:
            self.update(Panel("No data", title="Probe Details"))
            return

        if self.selected_token >= len(self.tokens):
            self.update(Panel("Invalid token", title="Probe Details"))
            return

        token = self.tokens[self.selected_token]

        # Create table of activations at this token
        table = RichTable(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Probe", style="cyan", width=28)
        table.add_column("Layers", justify="center", width=5)
        table.add_column("Conf", justify="right", width=7)
        table.add_column("Layer Bar", width=12)
        table.add_column("", width=2)

        # Collect activations at this position from ALL predictions
        # Count how many layers are ACTIVE at this specific token position
        activations = []
        max_layers_at_token = 0
        
        for pred in self.predictions:
            # For AggregatedPrediction, check activations across all layers at this token
            if hasattr(pred, 'layer_predictions'):
                # Count active layers at this token position
                # Only check layers that are in the aggregated prediction's active layers list
                active_layers_at_token = 0
                max_conf_at_token = 0.0
                any_active = False
                
                # Check if this is a sentiment prediction
                is_sentiment = pred.action_name == "sentiment"

                for layer_pred in pred.layer_predictions:
                    # Only consider layers that passed the threshold in aggregation
                    if layer_pred.layer in pred.layers:
                        if self.selected_token < len(layer_pred.token_activations):
                            tok_act = layer_pred.token_activations[self.selected_token]
                            conf_val = tok_act.confidence

                            # Check if this layer is active at this specific token
                            # For sentiment, use absolute value comparison
                            if is_sentiment:
                                if abs(conf_val) >= self.display_threshold:
                                    active_layers_at_token += 1
                                    any_active = True
                                # Track max by absolute value but keep sign
                                if abs(conf_val) > abs(max_conf_at_token):
                                    max_conf_at_token = conf_val
                            else:
                                if conf_val >= self.display_threshold:
                                    active_layers_at_token += 1
                                    any_active = True
                                max_conf_at_token = max(max_conf_at_token, conf_val)
                
                max_layers_at_token = max(max_layers_at_token, active_layers_at_token)
                
                # Add to activations list
                activations.append((pred.action_name, max_conf_at_token, any_active, active_layers_at_token))
            else:
                # Handle StreamingPrediction
                token_acts = pred.token_activations if hasattr(pred, 'token_activations') else []
                if self.selected_token < len(token_acts):
                    tok_act = token_acts[self.selected_token]
                    layer_count = 1
                    max_layers_at_token = max(max_layers_at_token, layer_count)
                    activations.append((pred.action_name, tok_act.confidence, tok_act.confidence >= self.display_threshold, layer_count))

        # Sort by layer count at this token (descending), then confidence
        # For sentiment (action name), sort by absolute value of confidence
        activations.sort(key=lambda x: (x[3], abs(x[1]) if x[0] == "sentiment" else x[1]), reverse=True)

        # Add ALL rows (not just top 15)
        for action, conf, is_active, layer_count_at_token in activations:
            # Special handling for sentiment probes (regression scores, not probabilities)
            if action == "sentiment":
                # Sentiment scores are z-score normalized (typically -3 to +3)
                # Use absolute value for color intensity
                abs_conf = abs(conf)
                # Normalize to 0-1 range for color (z-scores: 0=weak, 3=strong)
                normalized_for_color = min(abs_conf / 3.0, 1.0)
                color = confidence_to_color(normalized_for_color)
                conf_str = f"{conf:+6.2f}"  # e.g., "-1.43" or "+2.15"
            else:
                # Cognitive probes use 0-1 probabilities
                color = confidence_to_color(conf)
                conf_str = format_confidence(conf)

            # Bar represents layer count at this token position, not overall
            bar_len = int((layer_count_at_token / max(max_layers_at_token, 1)) * 12) if max_layers_at_token > 0 else 0
            bar = "█" * bar_len
            marker = "✓" if is_active else ""

            # Category color and tag
            category = get_action_category(action)
            cat_color = get_category_color(category)
            cat_tag = get_category_tag(category)

            action_short = truncate_text(action, 23)
            table.add_row(
                f"[{cat_color}]{action_short}[/{cat_color}] [{cat_color}]{cat_tag}[/{cat_color}]",
                f"[yellow]{layer_count_at_token}[/yellow]",
                f"[{color}]{conf_str}[/{color}]",
                f"[yellow]{bar}[/yellow]",
                f"[green]{marker}[/green]"
            )

        title_text = f"Token [{self.selected_token}]: '{token}' - All {len(activations)} Actions (sorted by layers)"
        self.update(Panel(table, title=title_text, border_style="green"))

    def watch_selected_token(self, new_value: int) -> None:
        """Called when selected_token changes"""
        self.refresh_details()


class ActivationHeatmap(Static):
    """Display heatmap of all activations across layers"""

    selected_token = reactive(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictions: Optional[List[AggregatedPrediction]] = None
        self.layer_range = (21, 30)  # Default layer range
        self.display_threshold: float = 0.1

    def set_data(self, predictions: List[AggregatedPrediction], layer_range: Tuple[int, int]):
        """Set data"""
        self.predictions = predictions
        self.layer_range = layer_range
        self.refresh_heatmap()

    def refresh_heatmap(self):
        """Refresh the heatmap display based on selected token"""
        if not self.predictions:
            self.update(Panel("No data", title="Layer Activation Heatmap"))
            return

        # Create heatmap showing ALL actions (rows) vs layers (columns)
        # Show activation at the SELECTED TOKEN position
        table = RichTable(show_header=True, header_style="bold cyan", box=None, padding=(0, 0))

        # Header: layer numbers
        table.add_column("Action", style="cyan", width=30, no_wrap=True)
        table.add_column("Ct", width=3, justify="center", style="yellow")  # Layer count
        for layer_idx in range(self.layer_range[0], self.layer_range[1] + 1):
            table.add_column(f"L{layer_idx}", width=3, justify="center", style="dim")

        # Group predictions by category
        grouped: Dict[str, List[AggregatedPrediction]] = {}
        for pred in self.predictions:
            cat = get_action_category(pred.action_name)
            grouped.setdefault(cat, []).append(pred)

        # Stable category order
        category_order = ["metacognitive", "analytical", "creative", "emotional", "memory", "other"]

        for category in category_order:
            preds_in_cat = grouped.get(category, [])
            if not preds_in_cat:
                continue

            # Sort within category by active layers at this token (desc)
            def active_layers_for(pred: AggregatedPrediction) -> int:
                count = 0
                is_sentiment = pred.action_name == "sentiment"
                for layer_idx in range(self.layer_range[0], self.layer_range[1] + 1):
                    if layer_idx not in pred.layers:
                        continue
                    layer_pred = next((p for p in pred.layer_predictions if p.layer == layer_idx), None)
                    if layer_pred and self.selected_token < len(layer_pred.token_activations):
                        conf = layer_pred.token_activations[self.selected_token].confidence
                        # For sentiment, check absolute value
                        if is_sentiment:
                            if abs(conf) >= self.display_threshold:
                                count += 1
                        else:
                            if conf >= self.display_threshold:
                                count += 1
                return count

            preds_in_cat.sort(key=lambda p: active_layers_for(p), reverse=True)

            # Category header row
            cat_color = get_category_color(category)
            cat_tag = get_category_tag(category)
            header_cells = [f"[{cat_color}]{category.upper()}[/{cat_color}] [{cat_color}]{cat_tag}[/{cat_color}]", "", *([""] * (self.layer_range[1] - self.layer_range[0] + 1))]
            table.add_row(*header_cells)

            # Rows for predictions in this category
            for pred in preds_in_cat:
                # Action cell with category styling
                action_cell = f"[{cat_color}]{truncate_text(pred.action_name, 25)}[/{cat_color}] [{cat_color}]{cat_tag}[/{cat_color}]"

                # Count active layers at this token
                active_layers_at_token = 0
                layer_cells = []
                is_sentiment = pred.action_name == "sentiment"

                for layer_idx in range(self.layer_range[0], self.layer_range[1] + 1):
                    layer_pred = next((p for p in pred.layer_predictions if p.layer == layer_idx), None)
                    if layer_pred and self.selected_token < len(layer_pred.token_activations):
                        tok_act = layer_pred.token_activations[self.selected_token]
                        conf = tok_act.confidence

                        # Special handling for sentiment (z-score normalized)
                        if is_sentiment:
                            # Normalize z-score to 0-1 for visualization (z-scores typically -3 to +3)
                            normalized_conf = min(abs(conf) / 3.0, 1.0)
                            symbol = get_activation_symbol(normalized_conf)
                            color = confidence_to_color(normalized_conf)
                            # Check if active using absolute value
                            if abs(conf) >= self.display_threshold and layer_idx in pred.layers:
                                active_layers_at_token += 1
                        else:
                            # Cognitive probes use 0-1 probabilities
                            symbol = get_activation_symbol(conf)
                            color = confidence_to_color(conf)
                            if conf >= self.display_threshold and layer_idx in pred.layers:
                                active_layers_at_token += 1

                        layer_cells.append(f"[{color}]{symbol}[/{color}]")
                    else:
                        layer_cells.append("[dim]·[/dim]")

                row = [action_cell, f"[yellow]{active_layers_at_token}[/yellow]"]
                row.extend(layer_cells)
                table.add_row(*row)

        title_text = f"Layer Activation at Token [{self.selected_token}] - All {len(self.predictions)} Actions"
        self.update(Panel(table, title=title_text, border_style="magenta"))

    def watch_selected_token(self, new_value: int) -> None:
        """Called when selected_token changes"""
        self.refresh_heatmap()


class SentimentTracker(Static):
    """Display sentiment analysis across tokens with sparkline visualization"""

    selected_token = reactive(0)

    def __init__(self, tokens: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = tokens
        self.predictions: Optional[List[AggregatedPrediction]] = None

    def set_predictions(self, predictions: List[AggregatedPrediction]):
        """Set predictions data"""
        self.predictions = predictions
        self.refresh()

    def render(self) -> Panel:
        """Render sentiment tracker with sparkline and current value"""
        try:
            if not self.predictions or not self.tokens:
                return Panel("No sentiment data", title="Sentiment Tracker", border_style="magenta", height=8)

            # Find sentiment predictions
            sentiment_preds = [p for p in self.predictions if p.action_name == "sentiment"]

            if not sentiment_preds:
                return Panel("No sentiment probes loaded", title="Sentiment Tracker 😊😐😢", border_style="magenta", height=8)

            # Get sentiment scores across all tokens from all layers
            num_tokens = len(self.tokens)
            sentiment_scores = [0.0] * num_tokens

            # Average sentiment across layers for each token
            for token_idx in range(num_tokens):
                scores_at_token = []
                for sent_pred in sentiment_preds:
                    for layer_pred in sent_pred.layer_predictions:
                        if token_idx < len(layer_pred.token_activations):
                            scores_at_token.append(layer_pred.token_activations[token_idx].confidence)
                if scores_at_token:
                    sentiment_scores[token_idx] = sum(scores_at_token) / len(scores_at_token)

            # Current token sentiment
            current_sentiment = sentiment_scores[self.selected_token] if self.selected_token < len(sentiment_scores) else 0.0

            # Determine sentiment label and emoji (z-scores: typically -3 to +3)
            # Use 0.5 as threshold since scores are normalized
            if current_sentiment > 0.5:
                sentiment_label = "POSITIVE"
                sentiment_emoji = "😊"
                sentiment_color = "green"
            elif current_sentiment < -0.5:
                sentiment_label = "NEGATIVE"
                sentiment_emoji = "😢"
                sentiment_color = "red"
            else:
                sentiment_label = "NEUTRAL"
                sentiment_emoji = "😐"
                sentiment_color = "yellow"

            # Create sparkline visualization with sliding window that follows the selected token
            sparkline_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            max_sparkline_width = 80

            # Use a sliding window approach that centers on the selected token
            if num_tokens > max_sparkline_width:
                # Calculate window bounds centered on selected token
                half_window = max_sparkline_width // 2
                window_start = max(0, self.selected_token - half_window)
                window_end = min(num_tokens, window_start + max_sparkline_width)

                # Adjust if we're at the end
                if window_end == num_tokens:
                    window_start = max(0, window_end - max_sparkline_width)

                # Extract window of scores
                windowed_scores = sentiment_scores[window_start:window_end]
                # Position of selected token within the window
                selected_in_window = self.selected_token - window_start

                # Show position indicator
                position_indicator = f" [{window_start+1}-{window_end}/{num_tokens}]"
            else:
                # Show all tokens if they fit
                windowed_scores = sentiment_scores
                selected_in_window = self.selected_token
                position_indicator = ""

            # Normalize z-scores to 0-1 range for sparkline (shift from [-3,3] to [0,1])
            # Z-scores typically range from -3 to +3
            normalized_scores = [(min(max(s, -3.0), 3.0) + 3.0) / 6.0 for s in windowed_scores]

            # Validate selected_in_window is within bounds
            if selected_in_window >= len(windowed_scores):
                selected_in_window = len(windowed_scores) - 1 if windowed_scores else 0

            # Build display
            text = Text()

            # Current sentiment (big display)
            text.append(f"\n  {sentiment_emoji}  Current: ", style="bold")
            text.append(f"{sentiment_label}", style=f"bold {sentiment_color}")
            text.append(f"  ({current_sentiment:+.2f})", style=sentiment_color)
            text.append("\n\n")

            # Sparkline with position indicator
            text.append("  Sentiment Flow", style="dim")
            if position_indicator:
                text.append(position_indicator, style="dim cyan")
            text.append(": ", style="dim")

            # Build sparkline directly into the Text object
            for idx, norm_score in enumerate(normalized_scores):
                # Defensive bounds check to prevent IndexError
                if idx >= len(windowed_scores):
                    break

                char_idx = min(int(norm_score * len(sparkline_chars)), len(sparkline_chars) - 1)
                char = sparkline_chars[char_idx]

                # Color based on sentiment (z-score thresholds)
                if windowed_scores[idx] > 0.5:
                    char_color = "green"
                elif windowed_scores[idx] < -0.5:
                    char_color = "red"
                else:
                    char_color = "yellow"

                # Highlight selected token with a marker
                if idx == selected_in_window:
                    text.append(char, style=f"bold {char_color} on white")
                else:
                    text.append(char, style=char_color)

            text.append("\n\n")

            # Summary stats
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            max_pos = max(sentiment_scores) if sentiment_scores else 0.0
            max_neg = min(sentiment_scores) if sentiment_scores else 0.0

            text.append("  Avg: ", style="dim")
            text.append(f"{avg_sentiment:+.2f}", style="cyan")
            text.append("  │  Peak: ", style="dim")
            text.append(f"{max_pos:+.2f}", style="green")
            text.append("  │  Low: ", style="dim")
            text.append(f"{max_neg:+.2f}", style="red")

            return Panel(text, title=f"Sentiment Tracker {sentiment_emoji}", border_style="magenta", height=8)

        except Exception as e:
            # If there's any error, return a clean error panel instead of crashing
            error_text = Text()
            error_text.append(f"\nError rendering sentiment: {str(e)}\n", style="red")
            error_text.append(f"\nPlease check that sentiment probes are loaded correctly.", style="dim")
            return Panel(error_text, title="Sentiment Tracker (Error)", border_style="red", height=8)

    def watch_selected_token(self, new_value: int) -> None:
        """Called when selected_token changes"""
        self.refresh()


class ProbeViewerApp(App):
    """Interactive TUI for exploring probe activations"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: auto auto 1fr;
    }

    #token-display {
        column-span: 2;
        height: 7;
        overflow-y: auto;
    }

    #sentiment-tracker {
        column-span: 2;
        height: 8;
    }

    #probe-details {
        height: 100%;
        overflow-y: auto;
    }

    #heatmap {
        height: 100%;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("left", "previous_token", "← Prev"),
        Binding("right", "next_token", "→ Next"),
        Binding("home", "first_token", "⇤ First"),
        Binding("end", "last_token", "⇥ Last"),
        Binding("r", "reload", "↻ Reload"),
    ]

    def __init__(
        self,
        predictions: List[AggregatedPrediction],
        tokens: List[str],
        layer_range: Tuple[int, int],
        display_threshold: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.predictions = predictions
        self.tokens = tokens
        self.layer_range = layer_range
        self.display_threshold = display_threshold
        self.selected_token = 0

    def compose(self) -> ComposeResult:
        """Create UI components"""
        yield Header()

        # Token display (top row, spans 2 columns)
        self.token_display = TokenDisplay(self.tokens, id="token-display")
        self.token_display.display_threshold = self.display_threshold
        self.token_display.set_predictions(self.predictions)
        yield self.token_display

        # Sentiment tracker (second row, spans 2 columns) - fills the gap!
        self.sentiment_tracker = SentimentTracker(self.tokens, id="sentiment-tracker")
        self.sentiment_tracker.set_predictions(self.predictions)
        yield self.sentiment_tracker

        # Probe details (left column, bottom)
        self.probe_details = ProbeDetails(id="probe-details")
        self.probe_details.display_threshold = self.display_threshold
        self.probe_details.set_data(self.predictions, self.tokens)
        yield self.probe_details

        # Heatmap (right column, bottom) - showing all layers
        self.heatmap = ActivationHeatmap(id="heatmap")
        self.heatmap.display_threshold = self.display_threshold
        self.heatmap.set_data(self.predictions, self.layer_range)
        yield self.heatmap

        yield Footer()

    def action_next_token(self) -> None:
        """Move to next token"""
        if self.selected_token < len(self.tokens) - 1:
            self.selected_token += 1
            self._update_selection()

    def action_previous_token(self) -> None:
        """Move to previous token"""
        if self.selected_token > 0:
            self.selected_token -= 1
            self._update_selection()

    def action_first_token(self) -> None:
        """Jump to first token"""
        self.selected_token = 0
        self._update_selection()

    def action_last_token(self) -> None:
        """Jump to last token"""
        self.selected_token = len(self.tokens) - 1
        self._update_selection()

    def action_reload(self) -> None:
        """Reload display"""
        self._update_selection()

    def _update_selection(self) -> None:
        """Update all panels with new selection"""
        self.token_display.selected_token = self.selected_token
        self.sentiment_tracker.selected_token = self.selected_token
        self.probe_details.selected_token = self.selected_token
        self.heatmap.selected_token = self.selected_token


def launch_interactive_viewer(
    predictions: List[AggregatedPrediction],
    tokens: List[str],
    layer_range: Tuple[int, int],
    display_threshold: float
):
    """
    Launch the interactive TUI

    Args:
        predictions: List of aggregated predictions sorted by layer count
        tokens: List of token strings
        layer_range: Tuple of (start_layer, end_layer)
    """
    app = ProbeViewerApp(predictions, tokens, layer_range, display_threshold)
    app.run()


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive probe viewer")
    parser.add_argument("--probes-dir", type=str, required=True, help="Probes directory")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="Model name")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--threshold", type=float, default=0.1, help="Activation threshold")
    parser.add_argument("--display-threshold", type=float, default=None, help="UI display threshold (defaults to --threshold)")

    args = parser.parse_args()

    # Initialize engine
    print("Loading model and probes...")
    engine = StreamingProbeInferenceEngine(
        probes_base_dir=Path(args.probes_dir),
        model_name=args.model,
        verbose=True
    )

    # Test text
    if args.text:
        text = args.text
    else:
        text = "After receiving feedback, I began reconsidering my approach and analyzing the problem."

    print(f"\nAnalyzing: '{text}'\n")

    # Run inference
    import time
    start = time.time()
    all_predictions = engine.predict_streaming(
        text,
        top_k=len(engine.probes),  # Get ALL predictions for aggregation
        threshold=0.0,  # Get all, filter during aggregation
        show_realtime=False  # Don't show streaming output
    )
    processing_time = time.time() - start

    # Aggregate predictions by action across layers
    aggregated_predictions = engine.aggregate_predictions(all_predictions, threshold=args.threshold)

    # Sort by layer count (descending), then by max confidence (descending)
    aggregated_predictions.sort(key=lambda x: (x.layer_count, x.max_confidence), reverse=True)

    # Get tokens
    inputs = engine.tokenizer(text, return_tensors="pt")
    tokens = engine.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    print(f"✓ Processed in {processing_time:.2f}s")
    print(f"✓ Found {len(aggregated_predictions)} aggregated actions across {len(all_predictions)} layer predictions")
    print(f"✓ Layer range: {engine.layer_start}-{engine.layer_end}")
    print(f"\nLaunching interactive viewer...\n")

    # Launch TUI with aggregated predictions sorted by layer count
    launch_interactive_viewer(
        aggregated_predictions, 
        tokens, 
        (engine.layer_start, engine.layer_end),
        args.display_threshold if args.display_threshold is not None else args.threshold
    )


if __name__ == "__main__":
    main()
